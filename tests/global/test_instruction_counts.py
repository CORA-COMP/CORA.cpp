"""test_instruction_counts - checks the parsing and the comparison of scripts/ci/instruction_counts.py.

Syntax:   python tests/global/test_instruction_counts.py
Checks:   the counts of cachegrind's and perf's output are read; growth above the threshold is
          reported, a decrease or growth below it is not.
Exit code: 0 if all checks pass, else 1.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "scripts", "ci"))
import instruction_counts as ic  # noqa: E402


class InstructionCountsTest(unittest.TestCase):
    def test_cachegrind_output(self):
        text = "==12== Collected : 1,234,567\n==12==\n==12== I   refs:      1,234,567\n"
        self.assertEqual(ic.parse_valgrind(text), 1234567)

    def test_perf_output(self):
        self.assertEqual(ic.parse_perf("4567890,,instructions:u,123,100.00,,\n"), 4567890)

    def test_unparsable_output_is_an_error(self):
        with self.assertRaises(ValueError):
            ic.parse_valgrind("nothing")
        with self.assertRaises(ValueError):
            ic.parse_perf("nothing")

    def test_growth_above_the_threshold_is_reported(self):
        _, grown = ic.compare({"a": 1000, "b": 1000, "c": 1000}, {"a": 1040, "b": 1020, "c": 900}, 3.0)
        self.assertEqual(grown, ["a"])

    def test_a_benchmark_new_in_the_head_is_reported(self):
        _, grown = ic.compare({}, {"a": 10}, 3.0)
        self.assertEqual(grown, ["a"])


if __name__ == "__main__":
    unittest.main()
