"""test_line_coverage - the reading, the union, the changed-line limit and the ratchet of lines.py

Syntax:   python tests/global/test_line_coverage.py
Checks:   gcov JSON gives executable and executed lines of src/ and include/ only; a line is executed
          if one suite ran it; the lines a branch adds are counted against the covered ones; the
          ratchet fails below the baseline and only raises it.
Exit code: 0 if all pass.

Authors:       Claude Sonnet 5.5
Written:       16-November-2026
Last update:   ---
Reviewed:      ---
"""
import argparse
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "scripts" / "coverage"))

import lines  # noqa: E402


class Lines(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.root = Path(self.dir.name).resolve()

    def tearDown(self):
        self.dir.cleanup()

    def test_only_the_library_is_read_and_documents_are_merged(self):
        # two documents of one file, one of a test and one outside the tree
        text = gcov(self.root, "src/a.cpp", {1: 1, 2: 0}) + gcov(self.root, "src/a.cpp", {2: 3, 4: 0}) \
            + gcov(self.root, "tests/t.cpp", {1: 1}) + gcov("/usr/include", "x.h", {1: 1})
        self.assertEqual(lines.read_gcov(text, self.root), {"src/a.cpp": ([1, 2, 4], [1, 2])})

    # the union of two suites
    def test_a_line_is_executed_if_any_suite_ran_it(self):
        union = lines.merge([{"src/a.cpp": ([1, 2], [1])}, {"src/a.cpp": ([1, 2, 3], [2])}])
        self.assertEqual(union, {"src/a.cpp": ([1, 2, 3], [1, 2])})
        # three lines are too few to count as a thin file, however many ran
        self.assertEqual(lines.totals(union), (2, 3, 0))

    def test_a_file_of_ten_lines_below_the_limit_is_thin(self):
        ten = list(range(1, 11))
        self.assertEqual(lines.totals({"src/a.cpp": (ten, ten[:6])}), (6, 10, 1))
        self.assertEqual(lines.totals({"src/a.cpp": (ten, ten[:7])}), (7, 10, 0))

    # a repository with a base commit and a head that adds two lines
    def test_the_lines_a_branch_adds_are_counted(self):
        (self.root / "src").mkdir()
        (self.root / "src" / "a.cpp").write_text("int a;\nint b;\n", encoding="utf-8")
        git(self.root, "init", "-q", "-b", "main")
        git(self.root, "add", "src/a.cpp")
        git(self.root, "commit", "-q", "-m", "base")
        (self.root / "src" / "a.cpp").write_text("int a;\nint b;\nint c;\nint d;\n", encoding="utf-8")
        git(self.root, "commit", "-q", "-am", "head")
        added = lines.added_lines("HEAD~1", self.root)
        self.assertEqual(added, {"src/a.cpp": {3, 4}})

    def test_the_ratchet_fails_below_the_baseline_and_only_rises(self):
        suites = {"unit": {"src/a.cpp": ([1, 2, 3, 4], [1, 2, 3])}}
        now = lines.baseline_numbers(suites)
        self.assertEqual(now["unit"], {"percent": 75.0, "thin_files": 0})
        self.assertEqual(now["combined"], now["unit"])
        # a baseline file, written by hand here, and compared with a run that is worse
        lines.BASELINE, saved = self.root / "lines.json", lines.BASELINE
        try:
            lines.BASELINE.write_text(json.dumps({"unit": {"percent": 80.0, "thin_files": 0}}), encoding="utf-8")
            data = self.root / "unit.json"
            data.write_text(json.dumps({"suite": "unit", "files": {"src/a.cpp": {
                "executable": [1, 2, 3, 4], "executed": [1, 2, 3]}}}), encoding="utf-8")
            self.assertEqual(lines.command_ratchet(argparse.Namespace(files=[str(data)], update=False)), 1)
            # the update never lowers it
            lines.command_ratchet(argparse.Namespace(files=[str(data)], update=True))
            self.assertEqual(json.loads(lines.BASELINE.read_text())["unit"]["percent"], 80.0)
        finally:
            lines.BASELINE = saved


# -----------------------------------------  AUXILIARY  ------------------------------------------ #

def gcov(root, name, counts):
    """One gcov document for the file `name` of `root` with {line: count}."""
    entry = {"file": f"{root}/{name}", "lines": [{"line_number": n, "count": c} for n, c in counts.items()]}
    return json.dumps({"current_working_directory": str(root), "files": [entry]})


def git(root, *args):
    subprocess.run(["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", *args], check=True,
                   capture_output=True)


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
