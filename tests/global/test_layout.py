"""test_layout - the layout rules of scripts/ci/layout.py: what they accept and what they refuse

Syntax:   python tests/global/test_layout.py
Checks:   on small fake files: a file in the layout passes; each of the docstring, author block,
          marker, comment and doc limits fails with a message that says what to do; a waiver
          with a reason excuses its lines and one without does not.
Exit code: 0 if all pass.

Authors:       Claude Sonnet 5.5
Written:       06-November-2026
Last update:   ---
Reviewed:      ---
"""
import sys
import unittest
from pathlib import Path

# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# -----------------------------------------  AUXILIARY  ------------------------------------------ #

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "ci"))
import layout  # noqa: E402


AUTHORS = ("// Authors:       Claude Sonnet 5.5\n// Written:       05-November-2026\n"
           "// Last update:   ---\n// Reviewed:      ---\n")
BEGIN, AUX, MAIN, END = (layout.marker(n) for n in ("BEGIN", "AUXILIARY", "MAIN", "END"))
CPP_TEST = (f"// test_plus - the sum\n//\n// Syntax:   plus(Z, W)\n// Inputs:   Z - a set\n//           W - another\n"
            f"//\n// CORA: contSet/@zonotope/plus.m\n// CORA authors: Mark Wetzlinger\n{AUTHORS}\n"
            f'#include "testing.h"\n\n{BEGIN}\n\n{AUX}\n\nnamespace {{\n// a helper\nvoid helper() {{}}\n'
            f"}} // namespace\n\n\n{MAIN}\n\nint main() {{\n    helper();\n    return 0;\n}}\n\n{END}\n")
PY_MODULE = ('"""plus - the sum\n\nAuthors:       Claude Sonnet 5.5\nWritten:       05-November-2026\n'
             'Last update:   06-November-2026\nReviewed:      06-November-2026 (Tobias Ladner)\n"""\n'
             f"import os\n\n{layout.marker('BEGIN', 'py')}\n\n\n{layout.marker('MAIN', 'py')}\n\n"
             f"def plus(a, b):\n    # the sum\n    return a + b\n\n{layout.marker('END', 'py')}\n")


def check(text, name="test_plus.cpp"):
    return layout.check_file(name, text.split("\n"))


# ============================================  MAIN  ============================================ #

class Accepts(unittest.TestCase):
    def test_a_file_in_the_layout_passes(self):
        self.assertEqual(check(CPP_TEST), [])
        self.assertEqual(check(PY_MODULE, "plus.py"), [])


class Docstring(unittest.TestCase):
    def fails(self, text, message, name="test_plus.cpp"):
        found = check(text, name)
        self.assertTrue(any(message in issue for issue in found), found)

    def test_the_first_line_names_the_file(self):
        self.fails(CPP_TEST.replace("test_plus - the sum", "the sum"), "should start with 'test_plus - ")

    def test_a_long_description_fails(self):
        prose = "".join(f"// line {k}\n" for k in range(13))
        self.fails(CPP_TEST.replace("//\n// CORA:", f"//\n{prose}//\n// CORA:", 1), "lines of prose")

    def test_an_entry_is_one_line(self):
        self.fails(CPP_TEST.replace("//           W - another", "//           W - another\n//             and more"),
                   "one line per entry")

    def test_two_entries_on_a_line_fail(self):
        self.fails(CPP_TEST.replace("Z - a set\n//           W - another", "Z - a set;  W - another"),
                   "two entries on one line")

    def test_a_syntax_form_is_one_line(self):
        self.fails(CPP_TEST.replace("plus(Z, W)", "plus(Z,\n//           W)"), "one line per entry")

    def test_the_author_block_is_required_and_formatted(self):
        self.fails(CPP_TEST.replace(AUTHORS, ""), "should end with the author block")
        self.fails(CPP_TEST.replace("Claude Sonnet 5.5", "Sonnet"), "lists the models")
        self.fails(CPP_TEST.replace("05-November-2026", "5 Nov 2026"), "'Written:' is a DD-Month-YYYY date")
        self.fails(CPP_TEST.replace("Last update:   ---", "Last update:   04-November-2026"), "earlier than")
        self.fails(CPP_TEST.replace("Reviewed:      ---", "Reviewed:      06-November-2026 (Claude Opus 5)"),
                   "records human reviews")
        self.fails(CPP_TEST.replace("Authors:       ", "Authors: "), "aligned to column 15")

    def test_a_ported_file_names_the_cora_authors(self):
        self.fails(CPP_TEST.replace("// CORA authors: Mark Wetzlinger\n", ""), "'CORA authors:'")

    def test_python_has_the_block_in_the_docstring(self):
        self.fails(PY_MODULE.replace("Authors:       Claude Sonnet 5.5\n", ""), "author block", "plus.py")


class Markers(unittest.TestCase):
    def fails(self, text, message, name="test_plus.cpp"):
        found = check(text, name)
        self.assertTrue(any(message in issue for issue in found), found)

    def test_every_marker_is_required(self):
        self.fails(CPP_TEST.replace(BEGIN, ""), "exactly one BEGIN")
        self.fails(CPP_TEST.replace(END, ""), "exactly one END")
        self.fails(CPP_TEST.replace(MAIN, ""), "exactly one MAIN")

    def test_auxiliary_code_needs_its_marker(self):
        self.fails(CPP_TEST.replace(AUX + "\n\n", ""), "needs the AUXILIARY marker")

    def test_an_empty_auxiliary_section_is_refused(self):
        self.fails(CPP_TEST.replace(f"namespace {{\n// a helper\nvoid helper() {{}}\n}} // namespace\n\n", "")
                   .replace("    helper();\n", ""), "without auxiliary code")

    def test_main_stands_above_main(self):
        self.fails(CPP_TEST.replace("int main", "int other;\nint main"), "between MAIN and `int main`")

    def test_main_has_two_empty_lines_above(self):
        self.fails(CPP_TEST.replace(f"\n\n\n{MAIN}", f"\n\n{MAIN}"), "two empty lines")

    def test_code_above_begin_fails(self):
        self.fails(CPP_TEST.replace('#include "testing.h"', "int x;"), "code above BEGIN")

    def test_a_header_has_no_main(self):
        header = f"// plus - the sum\n//\n{AUTHORS}\n#pragma once\n\n{BEGIN}\n\nint x;\n\n{END}\n"
        self.assertEqual(check(header, "plus.h"), [])
        self.fails(header.replace(f"int x;\n\n{END}", f"\n\n{MAIN}\n\nint x;\n\n{END}"), "unexpected MAIN", "plus.h")

    def test_python_main_stands_above_the_first_function(self):
        self.fails(PY_MODULE.replace("def plus", "def other():\n    return 1\n\n\ndef plus", 1)
                   .replace(layout.marker("MAIN", "py") + "\n\n", ""), "exactly one MAIN", "plus.py")
        moved = PY_MODULE.replace(layout.marker("MAIN", "py") + "\n\n", "").replace(
            "def plus", f"{layout.marker('MAIN', 'py')}\n\ndef plus", 1)
        self.assertEqual(check(moved, "plus.py"), [])


class Comments(unittest.TestCase):
    def fails(self, text, message, name="test_plus.cpp"):
        found = check(text, name)
        self.assertTrue(any(message in issue for issue in found), found)

    def test_a_run_of_code_needs_a_comment(self):
        def with_statements(count):
            return CPP_TEST.replace("    helper();\n", "".join(f"    x{k} = {k};\n" for k in range(count)))

        self.fails(with_statements(18), "without a comment")
        self.assertEqual(check(with_statements(17)), [])

    def test_a_block_comment_is_one_line(self):
        self.fails(CPP_TEST.replace("    helper();", "    // first\n    // second\n    helper();"), "block comment")

    def test_a_doc_comment_has_three_lines(self):
        self.fails(CPP_TEST.replace("void helper", "/// a\n/// b\n/// c\n/// d\nvoid helper"), "doc comment of 4")

    def test_python_docstrings_have_three_lines(self):
        self.fails(PY_MODULE.replace("    # the sum\n", '    """a\n    b\n    c\n    d"""\n'), "docstring of 4", "plus.py")


class Waivers(unittest.TestCase):
    def test_a_waiver_needs_a_reason(self):
        lines = ["a  // NOLINT(backend): why", "b  // NOLINT(backend)", "// NOLINTNEXTLINE(naming): why", "c",
                 "// NOLINTBEGIN(clones,backend): why", "d", "// NOLINTEND(clones,backend)", "e"]
        self.assertEqual(layout.waived(lines, "backend"), {0, 4, 5, 6})
        self.assertEqual(layout.waived(lines, "naming"), {3})


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
