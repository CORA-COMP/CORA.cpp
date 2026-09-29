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
                rows[name] = [float(v) for v in line.split()[1:4]]  # accuracy, then eps = 0.02, 0.05
    return rows


class SetBasedTraining(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rows = run_example()

    def test_both_models_learn_the_points(self):
        for name, row in self.rows.items():
            self.assertGreaterEqual(row[0], 0.95, name)

    def test_set_based_training_certifies_more_of_the_boxes(self):
        # eps = 0.05 is the size of the boxes the set-based model was trained with
        self.assertGreater(self.rows["set-based"][2], self.rows["standard"][2] + 0.05)


if __name__ == "__main__":
    unittest.main()
