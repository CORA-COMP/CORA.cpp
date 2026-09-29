"""test_neuralNetwork_train - set-based training certifies more of the training points than standard"""
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
EXAMPLE = os.path.join(ROOT, "examples", "python", "example_neuralNetwork_train_01_setBased.py")


def run_example():
    with tempfile.TemporaryDirectory() as folder:
        figure = os.path.join(folder, "train.png")
        result = subprocess.run([sys.executable, EXAMPLE, "--save", figure],
                                capture_output=True, text=True, cwd=ROOT)
        assert result.returncode == 0, result.stderr
        assert os.path.exists(figure)
    rows = {}
    for line in result.stdout.splitlines():
        for name in ("standard", "set-based"):
            if line.startswith(name + " "):
                rows[name] = [float(v) for v in line.split()[1:]]  # accuracy, then eps = 0.01, 0.02, 0.03
    return rows


class SetBasedTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = run_example()

    def test_both_models_learn_the_points(self):
        for name, row in self.rows.items():
            self.assertGreaterEqual(row[0], 0.95, name)

    def test_set_based_training_certifies_more_of_the_boxes(self):
        standard, setBased = self.rows["standard"], self.rows["set-based"]
        # eps = 0.02 and 0.03 are at and above the boxes the model was trained with
        self.assertGreater(setBased[2], standard[2] + 0.05)
        self.assertGreater(setBased[3], standard[3] + 0.05)


if __name__ == "__main__":
    unittest.main()
