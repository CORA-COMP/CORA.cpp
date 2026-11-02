"""test_coverage - the checker of scripts/coverage/coverage.py: what it accepts and what it fails

Syntax:   python tests/global/test_coverage.py
Checks:   on a small fake tree (two CORA methods, a source, a C++ and a Python test): the clean tree
          passes; a tag to a nonexistent row, a stale table, an operation without a tag, a status
          drop, a growing oracle debt, a test of an unimplemented row, a property test without a
          seed and an out-of-scope row that is implemented each fail with a clear message.
Exit code: 0 if all pass.
"""
import json
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "coverage"))
import coverage as cov  # noqa: E402

INVENTORY = {"cora_commit": "abc", "groups": {
    "contSet": {"zonotope": ["center", "plus"]}, "contDynamics": {}, "hybridDynamics": {},
    "nn": {}, "specification": {}, "others": {}}}
LEAN = {"coralean_commit": "def", "rows": {
    "contSet/zonotope/plus": {"status": "float", "decls": ["Zonotope.plus"]},
    "contSet/zonotope/center": {"status": "float", "decls": ["Zonotope.center"]}}}
SOURCE = "// plus - the sum\n//\n// Syntax: plus(Z, W)\n//\n// CORA: contSet/@zonotope/plus.m\n\nint x;\n"
CENTER = SOURCE.replace("plus", "center").replace("Z, W", "Z")
CPP_TEST = "// test_zonotope_plus - the sum\n//\n// CORA: contSet/@zonotope/plus.m\n\nint main() {}\n"
PY_TEST = '"""test_zonotope - the class"""\n# CORA: contSet/@zonotope/plus.m\n'
ORACLE_TEST = ("// test_lean_plus - against CORALean\n//\n// ORACLE: contSet/@zonotope/plus.m\n"
               "\nint main() {}\n")


def write(root, name, text):
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def make_tree(root, lean=True):
    """The fake tree of one tagged operation, with its tables generated."""
    write(root, "coverage/cora-inventory.json", json.dumps(INVENTORY))
    if lean:
        write(root, "coverage/coralean-status.json", json.dumps(LEAN))
    write(root, "README.md", f"# fake\n\n{cov.BEGIN}\n{cov.END}\n")
    write(root, "src/cont_set/zonotope/plus.cpp", SOURCE)
    write(root, "tests/cont_set/zonotope/test_zonotope_plus.cpp", CPP_TEST)
    write(root, "tests/python/test_zonotope.py", PY_TEST)
    write(root, "tests/global/oracle/test_lean_plus.cpp", ORACLE_TEST)
    problems, _ = cov.run(root, check=False)
    assert not problems, problems


class Checker(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        make_tree(self.root)

    def tearDown(self):
        self.dir.cleanup()

    def check(self, **kwargs):
        return cov.run(self.root, check=True, **kwargs)

    def assert_fails(self, text, **kwargs):
        problems, _ = self.check(**kwargs)
        self.assertTrue(any(text in p for p in problems), f"{text!r} not in {problems}")

    def test_a_clean_tree_passes_and_the_oracle_row_is_oracle(self):
        self.assertEqual(self.check(), ([], []))
        rows = json.loads((self.root / "coverage/status.json").read_text())["rows"]
        self.assertEqual(rows, {"contSet/zonotope/center": "todo", "contSet/zonotope/plus": "oracle"})

    def test_a_tag_naming_no_row_fails(self):
        text = SOURCE.replace("\nint x;", "// CORA: contSet/@zonotope/nope.m\n\nint x;")
        write(self.root, "src/cont_set/zonotope/plus.cpp", text)
        self.assert_fails("src/cont_set/zonotope/plus.cpp:6: `contSet/@zonotope/nope.m` names no inventory row")

    def test_a_tag_that_is_no_cora_file_fails(self):
        write(self.root, "tests/python/test_zonotope.py", PY_TEST + "# CORA: contSet/@zonotope/plus\n")
        self.assert_fails("is no CORA file")

    def test_a_stale_table_fails(self):
        table = self.root / "coverage/contSet/zonotope.md"
        table.write_text(table.read_text() + "\nedited by hand\n")
        self.assert_fails("generated files out of date")

    def test_a_stale_readme_number_fails(self):
        readme = self.root / "README.md"
        readme.write_text(readme.read_text().replace(cov.BEGIN, cov.BEGIN + "\n| 1 | 2 |"))
        self.assert_fails("generated files out of date")

    def test_an_operation_without_a_tag_fails(self):
        write(self.root, "src/cont_set/zonotope/center.cpp", "// center - the center\n\nint y;\n")
        self.assert_fails("src/cont_set/zonotope/center.cpp: names no CORA file")

    def test_a_reason_instead_of_a_tag_passes(self):
        write(self.root, "src/cont_set/zonotope/center.cpp", "// center - the center\n//\n"
              "// CORA.cpp only: batching\n\nint y;\n")
        self.assertEqual(self.check(), ([], []))

    def test_a_status_drop_fails_unless_accepted(self):
        (self.root / "tests/python/test_zonotope.py").write_text('"""test_zonotope - the class"""\n')
        (self.root / "tests/global/oracle/test_lean_plus.cpp").unlink()
        self.assert_fails("contSet/zonotope/plus: oracle -> untested: coverage may only go up")
        problems, warnings = self.check(accept_drop=["contSet/zonotope/plus"])
        self.assertFalse([p for p in problems if "may only go up" in p])
        self.assertIn("contSet/zonotope/plus: oracle -> untested: drop accepted", warnings)

    def test_writing_refuses_a_drop_and_records_an_accepted_one(self):
        (self.root / "tests/global/oracle/test_lean_plus.cpp").unlink()
        problems, _ = cov.run(self.root, check=False)
        self.assertTrue(any("may only go up" in p for p in problems))
        self.assertTrue(any("oracle debt 1" in p for p in problems))  # the drop makes debt
        problems, _ = cov.run(self.root, check=False, accept_drop=["contSet/zonotope/plus"],
                              accept_debt=True)
        self.assertEqual(problems, [])
        self.assertEqual(self.check(), ([], []))
        rows = json.loads((self.root / "coverage/status.json").read_text())["rows"]
        self.assertEqual(rows["contSet/zonotope/plus"], "ported")

    def test_growing_oracle_debt_fails_unless_accepted(self):
        write(self.root, "src/cont_set/zonotope/center.cpp", CENTER)
        write(self.root, "tests/cont_set/zonotope/test_zonotope_center.cpp",
              CPP_TEST.replace("plus", "center"))
        write(self.root, "tests/python/test_center.py",
              PY_TEST.replace("plus", "center").replace("test_zonotope", "test_center"))
        self.assert_fails("oracle debt 1 exceeds the committed 0")
        problems, warnings = self.check(accept_debt=True)
        self.assertFalse([p for p in problems if "oracle debt" in p])
        self.assertTrue(any("growth accepted" in w for w in warnings))

    def test_a_test_of_a_row_no_source_implements_fails(self):
        write(self.root, "tests/python/test_center.py",
              '"""test_center - center"""\n# CORA: contSet/@zonotope/center.m\n')
        self.assert_fails("tests `contSet/zonotope/center`, which no source file tags")

    def test_a_property_test_needs_a_fixed_seed(self):
        write(self.root, "tests/python/test_zonotope.py", PY_TEST + "# PROPERTY: contSet/@zonotope/plus.m\n")
        self.assert_fails("a PROPERTY test draws its inputs from a fixed seed")

    def test_an_oracle_test_lives_in_an_oracle_folder(self):
        write(self.root, "tests/python/test_zonotope.py", PY_TEST + "# ORACLE: contSet/@zonotope/plus.m\n")
        self.assert_fails("an ORACLE test is in a folder `oracle/`")

    def test_an_implemented_row_is_not_out_of_scope(self):
        write(self.root, "coverage/out-of-scope.md",
              "| Row | Reason |\n|---|---|\n| `contSet/zonotope/plus` | not needed |\n")
        self.assert_fails("`contSet/zonotope/plus` is implemented")

    def test_a_ledger_row_needs_a_reason_and_a_cora_row(self):
        write(self.root, "coverage/out-of-scope.md",
              "| Row | Reason |\n|---|---|\n| `contSet/zonotope/center` | |\n| `contSet/zonotope/x` | r |\n")
        self.assert_fails("`contSet/zonotope/center` needs a reason")
        self.assert_fails("`contSet/zonotope/x` is no CORA row")

    def test_a_missing_inventory_fails(self):
        (self.root / "coverage/cora-inventory.json").unlink()
        self.assert_fails("coverage/cora-inventory.json: missing")


if __name__ == "__main__":
    unittest.main()
