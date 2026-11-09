"""test_backing_report - the gcov parsing and the tag-to-row mapping of scripts/coverage/benchmarks.py

Syntax:   python tests/global/test_backing_report.py
Checks:   on a fixture of gcov JSON and a small fake tree: the files with an executed line are read
          (relative, only src/ and include/); a file's CORA tags give its rows and a `CORA.cpp only`
          file is listed apart; the counts per status come from the committed statuses; the report
          and the README table are rendered, spliced between their markers and stable on a second run.
Exit code: 0 if all pass.

Authors:       Claude Sonnet 5.5
Written:       09-November-2026
Last update:   ---
Reviewed:      ---
"""
import json
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
