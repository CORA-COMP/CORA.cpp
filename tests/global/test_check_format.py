"""test_check_format - checks the whitespace rules of scripts/ci/check_format.py.

Syntax:   python tests/global/test_check_format.py
Checks:   trailing whitespace, carriage returns, tabs in YAML and a missing final newline are
          reported with their line; clean text is not.
Exit code: 0 if all checks pass, else 1.

Authors:       Tobias Ladner
Written:       02-November-2026
Last update:   ---
Reviewed:      ---
"""
import os
import sys
import unittest


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "ci"))
import check_format as cf  # noqa: E402


# ============================================  MAIN  ============================================ #

class CheckFormatTest(unittest.TestCase):
    def test_clean_text(self):
        self.assertEqual(cf.problems("a.cpp", "int x;\n\n"), [])

    def test_trailing_whitespace(self):
        self.assertEqual(cf.problems("a.cpp", "ok\nbad \n"), ["a.cpp:2: trailing whitespace"])

    def test_carriage_return(self):
        self.assertEqual(cf.problems("a.cpp", "bad\r\n"), ["a.cpp:1: carriage return"])

    def test_tab_in_yaml_only(self):
        self.assertEqual(cf.problems("a.yml", "a:\n\tb\n"), ["a.yml:2: tab in YAML"])
        self.assertEqual(cf.problems("a.cpp", "a:\n\tb\n"), [])

    def test_final_newline(self):
        self.assertEqual(cf.problems("a.md", "text"), ["a.md: no final newline"])


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
