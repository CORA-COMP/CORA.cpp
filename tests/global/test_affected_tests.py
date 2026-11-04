"""test_affected_tests - checks the test selection of CI (scripts/ci/affected_tests.py) on synthetic trees.

Syntax:   python tests/global/test_affected_tests.py
Checks:   a changed header selects the tests that include it, directly or through other headers; a
          changed library source selects the tests of the headers it implements and what reaches
          them through private headers; the shared code, the build and CI select ALL; documentation
          selects NONE; unknown or deleted files select ALL.
Exit code: 0 if all checks pass, else 1.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "ci"))
import affected_tests as at  # noqa: E402

TREE = {
    "include/cora/cont_set/zonotope/zonotope.h": '#include "cora/global/rng.h"\n',
    "include/cora/cont_set/interval/interval.h": "",
    "include/cora/global/rng.h": "",
    "src/cont_set/zonotope/plus.cpp": '#include "cora/cont_set/zonotope/zonotope.h"\n#include "private/priv_sum.h"\n',
    "src/cont_set/zonotope/private/priv_sum.h": "",
    "src/cont_set/zonotope/private/priv_sum.cpp": '#include "priv_sum.h"\n',
    "src/cont_set/interval/plus.cpp": '#include "cora/cont_set/interval/interval.h"\n',
    "tests/testing.h": "",
    "tests/cont_set/zonotope/test_zonotope_plus.cpp":
        '#include "cora/cont_set/zonotope/zonotope.h"\n#include "testing.h"\n',
    "tests/cont_set/interval/test_interval_plus.cpp":
        '#include "cora/cont_set/interval/interval.h"\n#include "testing.h"\n',
    "tests/global/test_coding_conventions.py": "",
    "tests/global/test_other.py": "",
    "tests/python/test_zonotope.py": "",
    "examples/cpp/example_zonotope_01.cpp": '#include "cora/cont_set/zonotope/zonotope.h"\n',
}


def pick(*changed):
    result = at.select(list(changed), sorted(TREE), TREE.__getitem__)
    if result in ("ALL", "NONE"):
        return result
    return set(result[2:-2].split("|"))


class SelectionTest(unittest.TestCase):
    def test_header_selects_its_includers(self):
        self.assertEqual(pick("include/cora/cont_set/interval/interval.h"),
                         {"cont_set/interval/test_interval_plus", "conventions", "python", "install/consumer"})

    def test_header_reaches_tests_through_another_header(self):
        # rng.h is under include/cora/global: shared, so ALL
        self.assertEqual(pick("include/cora/global/rng.h"), "ALL")

    def test_source_selects_tests_of_its_header(self):
        self.assertEqual(pick("src/cont_set/zonotope/plus.cpp"),
                         {"cont_set/zonotope/test_zonotope_plus", "examples/example_zonotope_01", "conventions",
                          "python"})

    def test_private_source_is_reached_through_its_private_header(self):
        self.assertIn("cont_set/zonotope/test_zonotope_plus", pick("src/cont_set/zonotope/private/priv_sum.cpp"))
        self.assertNotIn("cont_set/interval/test_interval_plus", pick("src/cont_set/zonotope/private/priv_sum.cpp"))

    def test_a_test_selects_itself_only(self):
        self.assertEqual(pick("tests/cont_set/interval/test_interval_plus.cpp"),
                         {"cont_set/interval/test_interval_plus", "conventions"})
        self.assertEqual(pick("tests/global/test_other.py"), {"global/test_other", "conventions"})

    def test_testing_header_selects_every_cpp_test(self):
        self.assertEqual(pick("tests/testing.h"), "ALL")

    def test_python_tests_select_python(self):
        self.assertEqual(pick("tests/python/test_zonotope.py"), {"python", "conventions"})

    def test_shared_code_build_and_ci_select_all(self):
        for path in ("src/global/tensor/tensor.cpp", "src/global/backend/eigen.cpp", "CMakeLists.txt",
                     "scripts/cmake/python.cmake", "CMakePresets.json", ".github/workflows/ci.yml",
                     "scripts/ci/affected_tests.py"):
            self.assertEqual(pick(path), "ALL", path)

    def test_documentation_selects_nothing(self):
        self.assertEqual(pick("README.md", "docs/architecture.md"), "NONE")

    def test_unknown_and_deleted_files_select_all(self):
        self.assertEqual(pick("src/cont_set/zonotope/gone.cpp"), "ALL")
        self.assertEqual(pick("scripts/new_tool.py", "README.md"), "ALL")

    def test_documentation_does_not_hide_a_code_change(self):
        self.assertNotEqual(pick("README.md", "tests/cont_set/interval/test_interval_plus.cpp"), "NONE")

    def test_changes_are_resolved_against_the_real_tree(self):
        files = at.list_files()
        self.assertIn("conventions", at.tests_of(files))
        self.assertTrue(any(name.startswith("examples/") for name in at.tests_of(files)))


if __name__ == "__main__":
    unittest.main()
