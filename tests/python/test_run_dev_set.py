"""test_run_dev_set - a dev-set run whose tool deletes its result file (scripts/vnncomp/run_dev_set.py)

Authors:       Claude Opus 5.5
Written:       25-November-2026
Last update:   ---
Reviewed:      ---
"""
import argparse
import importlib.util
import os
import stat
import tempfile
import unittest


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_script():
    # the script by its path: it is no package
    path = os.path.join(ROOT, "scripts", "vnncomp", "run_dev_set.py")
    spec = importlib.util.spec_from_file_location("run_dev_set_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_tool(directory, run_body):
    # a tool interface of two scripts: prepare does nothing, run does run_body
    for name, body in (("prepare_instance.sh", "exit 0"), ("run_instance.sh", run_body)):
        path = os.path.join(directory, name)
        with open(path, "w") as stream:
            stream.write("#!/bin/bash\n" + body + "\n")
        os.chmod(path, os.stat(path).st_mode | stat.S_IXUSR)


class RunDevSetResultFile(unittest.TestCase):
    def setUp(self):
        # a fresh copy of the script, and a dev-set directory, a tool and a results directory
        self.script = load_script()
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = os.path.join(self.tmp.name, "bench")
        os.makedirs(os.path.join(self.dir, "1.0"))
        self.tool = os.path.join(self.tmp.name, "tool")
        os.makedirs(self.tool)
        self.results = os.path.join(self.tmp.name, "results")
        os.makedirs(self.results)

    def tearDown(self):
        self.tmp.cleanup()

    def run_row(self):
        # one instance of the dev set through the tool, as run_dev_set's pool runs it
        args = argparse.Namespace(dir=self.dir, version="1.0", max_timeout=0.0, results=self.results,
                                  threads=1, tool=self.tool)
        return self.script.run_one(args, 0, ("net.onnx", "prop.vnnlib", "5"))

    def test_a_deleted_result_file_is_no_result(self):
        # alpha-beta-CROWN's run_instance.sh removes the file first; an exit 0 without a result
        write_tool(self.tool, 'rm -f "$5"; exit 0')
        self.assertIn("no_result_in_file", self.run_row())

    def test_a_written_result_is_read(self):
        write_tool(self.tool, 'echo unsat > "$5"; exit 0')
        self.assertIn("unsat", self.run_row())


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
