"""test_run_instance - the verify options of a VNN-COMP instance run (scripts/vnncomp/run_instance.py)

Authors:       Claude Opus 5.5
Written:       23-November-2026
Last update:   ---
Reviewed:      ---
"""
import importlib.util
import os
import unittest
from unittest import mock

import cora


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# run_instance imports cora.nn_verify, which needs torch
try:
    import torch  # noqa: F401
except ImportError:
    torch = None

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def load_script():
    # the script by its path: it is no package
    path = os.path.join(ROOT, "scripts", "vnncomp", "run_instance.py")
    spec = importlib.util.spec_from_file_location("run_instance_script", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@unittest.skipUnless(torch is not None and cora.has_torch, "needs torch")
class RunInstanceOptions(unittest.TestCase):
    def setUp(self):
        # a fresh copy of the script per test
        self.script = load_script()

    def test_a_cuda_device_takes_lsncs_large_batch(self):
        with mock.patch.object(self.script, "batch_budget", return_value=1 << 30):
            options = self.script.options_for("lsnc_relu", "cuda")
        self.assertEqual(options.mini_batch_size, self.script.GPU_MINI_BATCH["lsnc_relu"])
        self.assertEqual(options.batch_memory_budget, 1 << 30)

    def test_the_cpu_and_other_categories_keep_the_preset(self):
        self.assertEqual(self.script.options_for("lsnc_relu", "cpu").mini_batch_size,
                         cora.vnncomp_options("lsnc_relu").mini_batch_size)
        with mock.patch.object(self.script, "batch_budget", return_value=1 << 30):
            options = self.script.options_for("acasxu", "cuda")
        self.assertEqual(options.mini_batch_size, cora.vnncomp_options("acasxu").mini_batch_size)

    def test_an_override_wins(self):
        env = {"CORACPP_VNNCOMP_OPTIONS": '{"mini_batch_size": 7}'}
        with mock.patch.dict(os.environ, env), \
                mock.patch.object(self.script, "batch_budget", return_value=1 << 30):
            self.assertEqual(self.script.options_for("lsnc_relu", "cuda").mini_batch_size, 7)


if __name__ == "__main__":
    unittest.main()

# ----------------------------------------  END OF CODE  ----------------------------------------- #
