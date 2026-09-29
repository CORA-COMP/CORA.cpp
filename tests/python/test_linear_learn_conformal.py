"""test_linear_learn_conformal - the conformal example covers the test trajectories as promised"""
import os
import re
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
EXAMPLE = os.path.join(ROOT, "examples", "python", "example_linear_learn_02_conformal.py")


def run_example():
    with tempfile.TemporaryDirectory() as folder:
        figure = os.path.join(folder, "conformal.png")
        result = subprocess.run([sys.executable, EXAMPLE, "--save", figure],
                                capture_output=True, text=True, cwd=ROOT)
        assert result.returncode == 0, result.stderr
        assert os.path.exists(figure)
    return result.stdout


class Conformal(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.output = run_example()

    def number(self, pattern):
        return float(re.search(pattern + r"\s*([0-9.]+)", self.output).group(1))

    def test_the_conformal_radius_reaches_the_promised_coverage(self):
        # 1000 test trajectories: a coverage a little under 0.9 is still within sampling error
        self.assertGreaterEqual(self.number("coverage with the conformal radius:"), 0.87)

    def test_the_reachable_set_alone_misses_trajectories(self):
        alone = self.number("coverage of the reachable set alone:")
        self.assertLess(alone, self.number("coverage with the conformal radius:") - 0.15)

    def test_the_radius_is_positive_and_small(self):
        radius = self.number("conformal radius:")
        self.assertGreater(radius, 0.0)
        self.assertLess(radius, 0.5)


if __name__ == "__main__":
    unittest.main()
