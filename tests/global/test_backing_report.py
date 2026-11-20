"""test_backing_report - the gcov parsing and the tag-to-row mapping of scripts/coverage/benchmarks.py

Syntax:   python tests/global/test_backing_report.py
Checks:   on a fixture of gcov JSON and a small fake tree: the files with an executed line are read
          (relative, only src/ and include/); a file's CORA tags give its rows and a `CORA.cpp only`
          file is listed apart; the counts per status come from the committed statuses; a row out of
          scope in CORALean, or not comparable with it, is in neither the numerator nor the
          denominator; the report
          and the README table are rendered, spliced between their markers and stable on a second run.
Exit code: 0 if all pass.

Authors:       Claude Sonnet 5.5, Claude Opus 5.5
Written:       09-November-2026
Last update:   19-November-2026
Reviewed:      ---
"""
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "coverage"))
import benchmarks as bench  # noqa: E402


INVENTORY = {"zonotope": ["center", "plus"]}
GROUPS = {"contSet": INVENTORY, "contDynamics": {}, "hybridDynamics": {}, "nn": {}, "specification": {},
          "others": {}}
STATUS = {"contSet/zonotope/plus": "oracle", "contSet/zonotope/center": "property"}
GCOV = json.dumps({"current_working_directory": "ROOT/build", "files": [
    {"file": "../src/plus.cpp", "lines": [{"line_number": 3, "count": 0}, {"line_number": 4, "count": 7}]},
    {"file": "../src/center.cpp", "lines": [{"line_number": 3, "count": 2}]},
    {"file": "../src/helper.cpp", "lines": [{"line_number": 3, "count": 1}]},
    {"file": "../src/never.cpp", "lines": [{"line_number": 3, "count": 0}]},
    {"file": "/usr/include/c++/vector", "lines": [{"line_number": 1, "count": 9}]}]})


def make_tree(root):
    """Sources: plus (one row), center (one row), helper (CORA.cpp only), never (not executed)."""
    for name, tag in [("plus", "// CORA: contSet/@zonotope/plus.m"),
                      ("center", "// CORA: contSet/@zonotope/center.m"),
                      ("helper", "// CORA.cpp only: a helper"), ("never", "// CORA: contSet/@zonotope/plus.m")]:
        (root / "src").mkdir(exist_ok=True)
        (root / "src" / f"{name}.cpp").write_text(f"// {name} - x\n//\n{tag}\n\nint x;\n", encoding="utf-8")


# ============================================  MAIN  ============================================ #

class BackingReport(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name)
        make_tree(self.root)

    def tearDown(self):
        self.dir.cleanup()

    def test_gcov_files_with_an_executed_line(self):
        text = GCOV.replace("ROOT", self.root.as_posix())
        # two documents in a row, as `gcov --stdout` prints them; outside src/ and include/ is ignored
        files = bench.parse_gcov(text + "\n" + text, self.root)
        self.assertEqual(files, ["src/center.cpp", "src/helper.cpp", "src/plus.cpp"])

    def test_tags_give_rows_and_only_files(self):
        rows, only = bench.rows_of_files(self.root, ["src/plus.cpp", "src/center.cpp", "src/helper.cpp"], GROUPS)
        self.assertEqual(rows, ["contSet/zonotope/center", "contSet/zonotope/plus"])
        self.assertEqual(only, ["src/helper.cpp"])

    def test_counts_use_the_committed_status(self):
        measured = {"A": {"files": ["src/plus.cpp", "src/center.cpp", "src/helper.cpp"]},
                    "B": {"skipped": "over 9 s"}}
        summary = bench.summarize(measured, STATUS, GROUPS, self.root)
        self.assertEqual((summary["A"]["oracle"], summary["A"]["total"]), (1, 2))
        self.assertEqual(summary["A"]["counts"]["property"], 1)
        self.assertEqual(summary["B"], {"skipped": "over 9 s"})

    # a driver includes a header that names every algorithm
    def test_a_header_counts_only_with_the_source_that_implements_it(self):
        # the header names plus and center; only plus.cpp ran, and center.cpp implements center
        (self.root / "include").mkdir()
        (self.root / "include" / "all.h").write_text(
            "// all - x\n//\n// CORA: contSet/@zonotope/plus.m\n// CORA: contSet/@zonotope/center.m\n\nint x;\n",
            encoding="utf-8")
        rows, _ = bench.rows_of_files(self.root, ["include/all.h", "src/plus.cpp"], GROUPS)
        self.assertEqual(rows, ["contSet/zonotope/plus"])

    # the denominator of the bit-exact count
    def test_rows_out_of_scope_in_CORALean_leave_the_denominator(self):
        measured = {"A": {"files": ["src/plus.cpp", "src/center.cpp"]}}
        lean = {"contSet/zonotope/center": {"status": "out-of-scope"}}
        s = bench.summarize(measured, STATUS, GROUPS, self.root, lean)["A"]
        self.assertEqual((s["oracle"], s["total"], s["lean_out"], s["eligible"]), (1, 2, 1, 1))
        self.assertIn("| A | 1/1 | 0 | 1 |", bench.render_table({"A": s}))

    # a bit-exact row that CORALean marks out of scope is in neither the numerator nor the denominator
    def test_a_bit_exact_row_out_of_scope_in_CORALean_leaves_the_numerator(self):
        measured = {"A": {"files": ["src/plus.cpp", "src/center.cpp"]}}
        lean = {"contSet/zonotope/plus": {"status": "out-of-scope"}}
        s = bench.summarize(measured, STATUS, GROUPS, self.root, lean)["A"]
        self.assertEqual((s["oracle"], s["lean_out"], s["eligible"]), (0, 1, 1))
        # a plumbing row (dim) at oracle, out of scope in CORALean: counted once, apart, never bit-exact
        status = {**STATUS, "contSet/zonotope/dim": "oracle"}
        groups = {**GROUPS, "contSet": {"zonotope": ["center", "plus", "dim"]}}
        (self.root / "src" / "dim.cpp").write_text(
            "// dim - x\n//\n// CORA: contSet/@zonotope/dim.m\n\nint x;\n", encoding="utf-8")
        lean = {"contSet/zonotope/dim": {"status": "out-of-scope"}}
        s = bench.summarize({"A": {"files": ["src/center.cpp", "src/dim.cpp"]}}, status, groups,
                            self.root, lean)["A"]
        self.assertEqual((s["oracle"], s["plumbing"], s["lean_out"], s["eligible"]), (0, 0, 1, 1))

    # a row of CORA.cpp's ledger that is not comparable with CORALean counts as one out of scope there
    def test_rows_not_comparable_with_CORALean_leave_the_denominator(self):
        measured = {"A": {"files": ["src/plus.cpp", "src/center.cpp"]}}
        apart = {"contSet/zonotope/plus": "no solver"}
        s = bench.summarize(measured, STATUS, GROUPS, self.root, {}, apart)["A"]
        self.assertEqual((s["oracle"], s["lean_out"], s["eligible"], s["property"]), (0, 1, 1, 1))

    # constructors and accessors have nothing to compare bit for bit
    def test_plumbing_rows_leave_the_denominator(self):
        for key in ["contSet/zonotope/zonotope", "contSet/contSet/contSet", "contSet/zonotope/dim",
                    "contSet/zonotope/generators", "contSet/polytope/isemptyobject",
                    "contSet/contSet/representsa_emptyObject", "contSet/contSet/reorder"]:
            self.assertTrue(bench.is_plumbing(key), key)
        for key in ["contSet/zonotope/plus", "contDynamics/linearSys/reach", "contSet/zonotope"]:
            self.assertFalse(bench.is_plumbing(key), key)
        # the bit-exact center is no plumbing; the plumbing row is neither in numerator nor denominator
        status = {**STATUS, "contSet/zonotope/zonotope": "oracle"}
        groups = {**GROUPS, "contSet": {"zonotope": ["center", "plus", "zonotope"]}}
        (self.root / "src" / "ctor.cpp").write_text(
            "// ctor - x\n//\n// CORA: contSet/@zonotope/zonotope.m\n\nint x;\n", encoding="utf-8")
        s = bench.summarize({"A": {"files": ["src/plus.cpp", "src/ctor.cpp"]}}, status, groups, self.root)["A"]
        self.assertEqual((s["total"], s["plumbing"], s["eligible"], s["oracle"]), (2, 1, 1, 1))

    # the column of the main tables is the summary, cell by cell
    def test_the_main_tables_are_filled_from_the_summary(self):
        measured = {"A": {"files": ["src/plus.cpp", "src/center.cpp"]}, "B-c": {"skipped": "x"}}
        summary = bench.summarize(measured, STATUS, GROUPS, self.root)
        readme = ("| instance | expected | oracle check (rows) | CORA [s] |\n|---|---|---|---|\n"
                  "| A | 1 | 9/9 bit-exact, 9 property | 5 |\n| B c | 1 | old | 6 |\n\ntext | x | y |\n")
        filled = bench.fill_main_tables(readme, summary).split("\n")
        self.assertEqual(filled[2], "| A | 1 | 1/2 bit-exact, 1 property | 5 |")
        self.assertEqual(filled[3], "| B c | 1 | old | 6 |")
        self.assertEqual(filled[5], "text | x | y |")
        self.assertIn("Over the 1 instances, 1 of 2 rows (50.0%)", bench.render_overall(summary))

    # a private step has no row, but it is listed
    def test_private_steps_are_listed(self):
        (self.root / "src" / "priv_x.cpp").write_text(
            "// priv_x - x\n//\n// CORA: contDynamics/@linearSys/private/priv_x.m\n\nint x;\n", encoding="utf-8")
        s = bench.summarize({"A": {"files": ["src/priv_x.cpp"]}}, STATUS, GROUPS, self.root)["A"]
        self.assertEqual(s["steps"], ["contDynamics/@linearSys/private/priv_x.m"])

    # a PR that deletes or renames a measured source must say which instances to measure again
    def test_a_missing_file_names_the_instances_to_measure_again(self):
        measured = {"A": {"files": ["src/plus.cpp", "src/gone.cpp"]}, "B": {"files": ["src/center.cpp"]},
                    "C": {"skipped": "x"}, "D": {"files": ["src/old.cpp", "src/gone.cpp"]}}
        text = "\n".join(bench.missing_files(self.root, measured))
        self.assertIn("A: src/gone.cpp no longer exists", text)
        self.assertIn("D: src/old.cpp no longer exists", text)
        self.assertNotIn("B:", text)
        self.assertIn("benchmarks.py --only A --only D", text)
        self.assertEqual(bench.missing_files(self.root, {"B": measured["B"], "C": measured["C"]}), [])

    def test_files_flag_fails_on_a_benchmarks_json_with_a_missing_file(self):
        (self.root / "coverage").mkdir()
        (self.root / "coverage" / "benchmarks.json").write_text(
            json.dumps({"instances": {"A": {"files": ["src/gone.cpp"]}}}), encoding="utf-8")
        script = ROOT / "scripts" / "coverage" / "benchmarks.py"
        # a copy of the script under the fake root: REPO is the root two levels above it
        (self.root / "scripts" / "coverage").mkdir(parents=True)
        for source in script.parent.glob("*.py"):
            (self.root / "scripts" / "coverage" / source.name).write_text(
                source.read_text(encoding="utf-8"), encoding="utf-8")
        run = subprocess.run([sys.executable, str(self.root / "scripts" / "coverage" / "benchmarks.py"), "--files"],
                             capture_output=True, text=True)
        self.assertEqual(run.returncode, 1, run.stderr)
        self.assertIn("A: src/gone.cpp no longer exists", run.stdout, run.stderr)
        self.assertIn("benchmarks.py --only A", run.stdout)

    def test_report_and_table_are_rendered_and_spliced(self):
        measured = {"A": {"files": ["src/plus.cpp", "src/center.cpp"]}}
        summary = bench.summarize(measured, STATUS, GROUPS, self.root)
        markdown = bench.render_markdown(summary, {"contSet/zonotope/center": {"status": "float"}})
        self.assertIn("| A | 2 | 1 | 1 |", markdown)
        self.assertIn("| `contSet/zonotope/center` | property | float | 1 | A |", markdown)
        # the table of the README, and the markers around it
        table = bench.render_table(summary)
        self.assertIn("| A | 1/2 | 1 |", table)
        once = bench.splice("# Title\n", table)
        self.assertEqual(bench.splice(once, table), once)
        self.assertIn(bench.BEGIN, once)
        # a changed table replaces the old one and keeps the text around it
        other = bench.splice(once + "\nafter\n", "| x |")
        self.assertTrue(other.startswith("# Title") and other.endswith("\nafter\n"))
        self.assertNotIn("| A | 1/2 |", other)


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
