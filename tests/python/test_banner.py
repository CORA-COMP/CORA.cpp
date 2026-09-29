"""test_banner - the CORA START and CORA END blocks of the Python examples"""
import os
import subprocess
import sys
import tempfile
import unittest

import cora

PACKAGE_PARENT = os.path.dirname(os.path.dirname(os.path.abspath(cora.__file__)))


def run(name, code):
    """Runs a script called `name` that imports cora and then executes `code`; returns the result."""
    with tempfile.TemporaryDirectory() as folder:
        path = os.path.join(folder, name)
        with open(path, "w", encoding="utf-8") as f:
            f.write("import cora\n" + code + "\n")
        env = dict(os.environ, PYTHONPATH=PACKAGE_PARENT)
        return subprocess.run([sys.executable, path], capture_output=True, text=True, env=env)


class Banner(unittest.TestCase):
    def test_an_example_prints_header_and_footer(self):
        out = run("example_banner_check.py", "print('body')").stdout
        self.assertIn("CORA START", out)
        self.assertIn(f"CORA.cpp | example_banner_check | default backend: {cora.backend()}", out)
        self.assertIn("body", out)
        self.assertIn("CORA END", out)
        self.assertIn("CORA.cpp | example_banner_check | runtime: ", out)
        self.assertLess(out.index("CORA START"), out.index("body"))
        self.assertLess(out.index("body"), out.index("CORA END"))

    def test_the_blocks_are_as_wide_as_the_cpp_ones(self):
        lines = run("example_banner_check.py", "").stdout.splitlines()
        self.assertTrue(all(len(line) == 70 for line in lines if line.startswith("=")))

    def test_another_script_prints_nothing(self):
        self.assertEqual(run("analysis.py", "print('only this')").stdout, "only this\n")

    def test_an_example_that_fails_prints_no_footer(self):
        result = run("example_banner_fails.py", "raise RuntimeError('broken on purpose')")
        self.assertIn("CORA START", result.stdout)
        self.assertNotIn("CORA END", result.stdout)
        self.assertIn("broken on purpose", result.stderr)


if __name__ == "__main__":
    unittest.main()
