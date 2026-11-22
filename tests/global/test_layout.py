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
        # a syntax form is one line
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
        # exactly one MAIN
        self.fails(CPP_TEST.replace(MAIN, ""), "exactly one MAIN")

    def test_auxiliary_code_needs_its_marker(self):
        self.fails(CPP_TEST.replace(AUX + "\n\n", ""), "needs the AUXILIARY marker")

    def test_an_empty_auxiliary_section_is_refused(self):
        self.fails(CPP_TEST.replace("namespace {\n// a helper\nvoid helper() {}\n} // namespace\n\n", "")
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

    def test_python_has_the_operation_first_and_no_main(self):
        # the operation right under BEGIN CODE, the helpers under AUXILIARY, no MAIN marker
        self.fails(PY_MODULE.replace(f"\n\n\n{PY_AUX}", f"\n\n\n{PY_MAIN}\n\n\n{PY_AUX}"), "unexpected MAIN", "plus.py")
        # the operation comes first, so an AUXILIARY marker right under BEGIN CODE is refused
        self.fails(PY_MODULE.replace(f"{PY_BEGIN}\n\ndef plus", f"{PY_BEGIN}\n\n{PY_AUX}\n\ndef plus"),
                   "right under BEGIN CODE", "plus.py")
        # AUXILIARY has two empty lines above it, and code below it
        self.fails(PY_MODULE.replace(f"\n\n\n{PY_AUX}", f"\n\n{PY_AUX}"), "two empty lines", "plus.py")
        self.fails(PY_MODULE.replace("def helper(a):\n    # the helper\n    return a\n", ""),
                   "without auxiliary code", "plus.py")

    # a file without helpers has no AUXILIARY marker
    def test_python_without_helpers_has_no_auxiliary_marker(self):
        alone = PY_MODULE.replace(f"{PY_AUX}\n\ndef helper(a):\n    # the helper\n    return a\n\n", "")
        self.assertEqual(check(alone.replace("helper(a)", "a"), "plus.py"), [])


class Imports(unittest.TestCase):
    def test_examples_use_from_cora_import(self):
        text = PY_MODULE.replace("import os", "import cora")
        text = text.replace("plus - the sum", "example_plus - the sum")
        found = check(text, "examples/python/example_plus.py")
        self.assertTrue(any("from cora import" in issue for issue in found), found)
        self.assertEqual(check(text.replace("import cora", "from cora import eye"),
                               "examples/python/example_plus.py"), [])

    def test_tests_and_library_use_import_cora(self):
        text = PY_MODULE.replace("import os", "from cora import eye")
        for name in ("tests/python/plus.py", "src/global/python/cora/plus.py", "scripts/plus.py"):
            found = check(text, name)
            self.assertTrue(any("import cora" in issue for issue in found), found)
        self.assertEqual(check(PY_MODULE.replace("import os", "import cora"), "tests/python/plus.py"), [])


# the comment rules
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
        docstring = '    """a\n    b\n    c\n    d"""\n'
        self.fails(PY_MODULE.replace("    # the sum\n", docstring), "docstring of 4", "plus.py")


ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(ROOT / "scripts" / "ci"))

import layout  # noqa: E402


# shell, CMake and .def files
SH = ("#!/usr/bin/env bash\n# run - runs a thing\n#\n# Authors:       Claude Sonnet 5.5\n"
      "# Written:       01-October-2026\n# Last update:   ---\n# Reviewed:      ---\n\n"
      + layout.marker("BEGIN", "py") + "\n\nset -e\necho hi\n\n" + layout.marker("END", "py") + "\n")


class OtherFiles(unittest.TestCase):
    # the problems of a file, one of which must contain the message
    def fails(self, text, message, name="scripts/run.sh"):
        found = layout.check_other_file(name, text.split("\n"))
        self.assertTrue(any(message in issue for issue in found), found)

    def test_a_file_in_the_layout_passes(self):
        self.assertEqual(layout.check_other_file("scripts/run.sh", SH.split("\n")), [])
        cmake = SH.replace("#!/usr/bin/env bash\n", "").replace("run - runs", "run - sets")
        self.assertEqual(layout.check_other_file("scripts/cmake/run.cmake", cmake.split("\n")), [])
        # CMakeLists.txt is named by its stem
        text = cmake.replace("run - sets", "CMakeLists - sets")
        self.assertEqual(layout.check_other_file("CMakeLists.txt", text.split("\n")), [])

    def test_the_docstring_and_the_author_block(self):
        self.fails(SH.replace("# run - runs a thing", "# runs a thing"), "should start with 'run - ")
        self.fails(SH.replace("# Authors:       Claude Sonnet 5.5\n", ""), "author block")

    # the markers: one END, no code above BEGIN
    def test_the_markers(self):
        self.fails(SH.replace(layout.marker("END", "py") + "\n", ""), "exactly one END")
        self.fails(SH.replace("\n\n" + layout.marker("BEGIN", "py"), "\nset -u\n\n" + layout.marker("BEGIN", "py")),
                   "code above BEGIN CODE")

    def test_a_run_of_code_needs_a_comment(self):
        many = "".join(f"echo {k}\n" for k in range(21))
        self.fails(SH.replace("echo hi\n", many), "without a comment")
        self.assertEqual(layout.check_other_file("scripts/run.sh", SH.replace(
            "echo hi\n", many.replace("echo 10\n", "# half\necho 10\n")).split("\n")), [])

    # the rows of a table are its content
    def test_a_def_table_has_no_density_rule(self):
        rows = "".join(f"ROW({k})\n" for k in range(40))
        text = ("// t - a table\n//\n// Authors:       Claude Sonnet 5.5\n// Written:       01-October-2026\n"
                "// Last update:   ---\n// Reviewed:      ---\n\n" + layout.marker("BEGIN", "cpp") + "\n\n" + rows
                + "\n" + layout.marker("END", "cpp") + "\n")
        self.assertEqual(layout.check_other_file("include/t.def", text.split("\n")), [])


class Waivers(unittest.TestCase):
    def test_a_waiver_needs_a_reason(self):
        # waivers need reasons
        lines = ["a  // NOLINT(backend): why", "b  // NOLINT(backend)", "// NOLINTNEXTLINE(naming): why", "c",
                 "// NOLINTBEGIN(clones,backend): why", "d", "// NOLINTEND(clones,backend)", "e"]
        self.assertEqual(layout.waived(lines, "backend"), {0, 4, 5, 6})
        self.assertEqual(layout.waived(lines, "naming"), {3})


# -----------------------------------------  AUXILIARY  ------------------------------------------ #

AUTHORS = ("// Authors:       Claude Sonnet 5.5\n// Written:       05-November-2026\n"
           "// Last update:   ---\n// Reviewed:      ---\n")

BEGIN, AUX, MAIN, END = (layout.marker(n) for n in ("BEGIN", "AUXILIARY", "MAIN", "END"))
PY_BEGIN, PY_AUX, PY_MAIN, PY_END = (layout.marker(n, "py") for n in ("BEGIN", "AUXILIARY", "MAIN", "END"))

CPP_TEST = (f"// test_plus - the sum\n//\n// Syntax:   plus(Z, W)\n// Inputs:   Z - a set\n//           W - another\n"
            f"//\n// CORA: contSet/@zonotope/plus.m\n// CORA authors: Mark Wetzlinger\n{AUTHORS}\n"
            f'#include "testing.h"\n\n{BEGIN}\n\n{AUX}\n\nnamespace {{\n// a helper\nvoid helper() {{}}\n'
            f"}} // namespace\n\n\n{MAIN}\n\nint main() {{\n    helper();\n    return 0;\n}}\n\n{END}\n")

PY_MODULE = ('"""plus - the sum\n\nAuthors:       Claude Sonnet 5.5\nWritten:       05-November-2026\n'
             'Last update:   06-November-2026\nReviewed:      06-November-2026 (Tobias Ladner)\n"""\n'
             f"import os\n\n{PY_BEGIN}\n\ndef plus(a, b):\n    # the sum\n    return helper(a) + b\n\n\n"
             f"{PY_AUX}\n\ndef helper(a):\n    # the helper\n    return a\n\n{PY_END}\n")


def check(text, name="test_plus.cpp"):
    return layout.check_file(name, text.split("\n"))


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
