"""test_linear_learn - the learning example finds the system matrix behind the measurements"""
import ast
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
EXAMPLE = os.path.join(ROOT, "examples", "python", "example_linear_learn_01_dynamics.py")


class LearnDynamics(unittest.TestCase):
    def test_the_learned_matrix_is_close_to_the_true_one(self):
        with tempfile.TemporaryDirectory() as folder:
            figure = os.path.join(folder, "learned.png")
            result = subprocess.run([sys.executable, EXAMPLE, "--save", figure],
                                    capture_output=True, text=True, cwd=ROOT)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(os.path.exists(figure))
        lines = result.stdout.splitlines()
        learned = ast.literal_eval(next(l for l in lines if l.startswith("learned A:"))[10:].strip())
        for row, expected in zip(learned, [[-0.1, 1.0], [-1.0, -0.1]]):
            for value, target in zip(row, expected):
                self.assertAlmostEqual(value, target, delta=0.1)

    def test_the_measurements_end_up_inside(self):
        result = subprocess.run([sys.executable, EXAMPLE, "--save", os.devnull],
                                capture_output=True, text=True, cwd=ROOT)
        last = [l for l in result.stdout.splitlines() if l.startswith("step 300")][0]
        outside = float(last.split("outside")[1].split(",")[0])
        self.assertLess(outside, 0.01)


if __name__ == "__main__":
    unittest.main()
