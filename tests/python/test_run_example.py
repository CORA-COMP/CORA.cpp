"""test_run_example - scripts/run_example.py, the F5 of VS Code, runs Python and C++ examples"""
import os
import subprocess
import sys
import tempfile
import unittest

ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SCRIPT = os.path.join(ROOT, "scripts", "run_example.py")


def run(path):
    """Runs the script on the file `path` and returns the finished process."""
    return subprocess.run([sys.executable, SCRIPT, path], capture_output=True, text=True, cwd=ROOT)


class RunExample(unittest.TestCase):
    def test_a_python_example_runs_with_its_blocks(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "example_run_check.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write("import cora\nprint('example body')\n")
            result = run(path)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("CORA START", result.stdout)
        self.assertIn("CORA.cpp | example_run_check.py | default backend:", result.stdout)
        self.assertIn("example body", result.stdout)
        self.assertIn("CORA END", result.stdout)

    def test_a_failing_python_example_fails(self):
        with tempfile.TemporaryDirectory() as folder:
            path = os.path.join(folder, "example_run_fails.py")
            with open(path, "w", encoding="utf-8") as f:
                f.write("import cora\nraise RuntimeError('broken on purpose')\n")
            result = run(path)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("broken on purpose", result.stderr)

    def test_something_that_is_no_example_is_described(self):
        result = run(os.path.join(ROOT, "README.md"))
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("neither a .py nor a .cpp example", result.stderr)

    def test_a_cpp_example_is_built_and_run(self):
        result = run(os.path.join(ROOT, "examples", "cpp", "example_linear_reach_03_specification.cpp"))
        self.assertEqual(result.returncode, 0, result.stderr)
        header = "CORA.cpp | example_linear_reach_03_specification.cpp | default backend:"
        self.assertIn(header, result.stdout)
        self.assertIn("CORA END", result.stdout)


if __name__ == "__main__":
    unittest.main()
