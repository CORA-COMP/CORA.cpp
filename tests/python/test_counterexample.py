"""test_counterexample - counterexamples written for VNN-COMP and checked as its scoring checks them

A small network and a property, in VNN-LIB 1.0 and 2.0: a valid counterexample is CORRECT, one
whose output misses the unsafe set is SPEC_NOT_VIOLATED, one 5e-5 outside the input box is
CORRECT_UP_TO_TOLERANCE and 1e-2 outside SPEC_NOT_VIOLATED; a float32 input on the bound rounded
to float32 passes (0.7 rounds down: the scoring compares in float32); malformed and empty texts. With
VNNCOMP_SCORING set to vnncomp2026_results/SCORING (and its requirements installed), every case is
also checked by the official code, which must agree.

Authors:       Claude Opus 5.5
Written:       21-November-2026
Last update:   ---
Reviewed:      ---
"""

import contextlib
import os
import pathlib
import sys
import tempfile
import unittest


try:
    import numpy as np
    import onnx
    import onnxruntime  # noqa: F401
    import vnnlib  # noqa: F401
    from onnx import TensorProto, helper, numpy_helper
except ImportError:
    raise unittest.SkipTest("needs onnx, onnxruntime and vnnlib") from None

import cora.counterexample as ce
from cora.vnnlib import parse_vnnlib, to_vnnlib2


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

class Check(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)
        for sub in ("onnx", "vnnlib"):
            os.makedirs(os.path.join(self.dir.name, sub))
        self.onnx = os.path.join(self.dir.name, "onnx", "net.onnx")
        network(self.onnx)
        self.count = 0

    def check(self, x, threshold, expected, lo=0.0, hi=1.0):
        """The verdict on x in both versions, against the expected one (and the official code)."""
        text1 = prop_text(lo, hi, threshold)
        for version, text in ((1, text1), (2, to_vnnlib2(text1, (1, 2), (1, 2)))):
            prop = parse_vnnlib(text, 2, 2)
            content = ce.write_counterexample(np.asarray(x, np.float32), output(x), prop)
            self.count += 1
            path = os.path.join(self.dir.name, "vnnlib", f"p{self.count}.vnnlib")
            with open(path, "w") as out:
                out.write(text)
            result, why = ce.check_counterexample(self.onnx, path, content)
            self.assertEqual(result, expected, f"version {version}: {why}")
            self.official(version, path, content, expected)

    def official(self, version, path, content, expected):
        """The official scoring's verdict on the same files, where it is available."""
        with official_scoring() as modules:
            if modules is None:
                return
            # the counterexample as a file, read by the 1.0 or the 2.0 validator
            ce_path = path + ".counterexample"
            with open(ce_path, "w") as out:
                out.write(content)
            if version == 1:
                result, _ = modules[0].get_ce_diff(self.onnx, path, ce_path, ce.COUNTEREXAMPLE_ATOL, 0.0)
            else:
                result, _ = modules[1].validate_vnnlib2_counterexample(
                    pathlib.Path(self.dir.name), "onnx/net.onnx", "vnnlib/" + os.path.basename(path),
                    ce_path, ce.COUNTEREXAMPLE_ATOL, 0.0, modules[0].CounterexampleResult, True)
            self.assertEqual(result, expected, f"official, version {version}")

    def test_valid(self):
        x = [0.5, 0.25]
        self.check(x, float(output(x)[0]) - 0.125, ce.CORRECT)

    def test_on_the_threshold(self):
        # y_0 equal to the threshold violates (>=), tolerance 0
        x = [0.5, 0.25]
        self.check(x, float(output(x)[0]), ce.CORRECT)

    def test_output_misses(self):
        x = [0.5, 0.25]
        self.check(x, float(output(x)[0]) + 0.125, ce.SPEC_NOT_VIOLATED)

    def test_input_tolerance(self):
        # 5e-5 outside the box is within the scoring's 1e-4, 1e-2 is not
        self.check([1.0 + 5e-5, 0.25], 0.0, ce.CORRECT_UP_TO_TOLERANCE)
        self.check([1.0 + 1e-2, 0.25], 0.0, ce.SPEC_NOT_VIOLATED)

    def test_bound_rounded_to_float32(self):
        # 0.7 rounds down to the float32 0.69999998...: that point counts as inside, below it not
        lo = 0.7
        x32 = float(np.float32(lo))
        self.assertLess(x32, lo)
        self.check([x32, 0.8], 0.0, ce.CORRECT, lo=lo)
        below = float(np.nextafter(np.float32(lo), np.float32(0)))
        self.check([below, 0.8], 0.0, ce.CORRECT_UP_TO_TOLERANCE, lo=lo)

    def test_malformed_and_missing(self):
        path = os.path.join(self.dir.name, "vnnlib", "p.vnnlib")
        with open(path, "w") as out:
            out.write(prop_text(0.0, 1.0, 0.0))
        self.assertEqual(ce.check_counterexample(self.onnx, path, "")[0], ce.NO_CE)
        self.assertEqual(ce.check_counterexample(self.onnx, path, "((X_1 0.5))")[0], ce.MALFORMED_CE)
        v2 = os.path.join(self.dir.name, "vnnlib", "p2.vnnlib")
        with open(v2, "w") as out:
            out.write(to_vnnlib2(prop_text(0.0, 1.0, 0.0), (1, 2), (1, 2)))
        self.assertEqual(
            ce.check_counterexample(self.onnx, v2, "X float32 [1, 2]\n0.5\n")[0], ce.MALFORMED_CE
        )


class Write(unittest.TestCase):
    def test_formats(self):
        # 1.0: (X_i v) and (Y_j v); 2.0: a header per variable and one value per line
        text1 = prop_text(0.0, 1.0, 0.0)
        x, y = np.array([0.1, 0.2], np.float32), np.array([1.5, 0.0], np.float32)
        v1 = ce.write_counterexample(x, y, parse_vnnlib(text1, 2, 2))
        self.assertTrue(v1.startswith("((X_0 0.10000000149011612)\n(X_1 "))
        v2 = ce.write_counterexample(x, y, parse_vnnlib(to_vnnlib2(text1, (1, 2), (1, 2)), 2, 2))
        self.assertEqual(v2.splitlines()[0], "X float32 [1, 2]")
        self.assertEqual(v2.splitlines()[3], "Y float32 [1, 2]")
        # every value parses back to the same float32
        values = [float(v.split()[1].rstrip(")")) for v in v1.splitlines()]
        np.testing.assert_array_equal(np.float32(values), np.concatenate([x, y]))


# -----------------------------------------  AUXILIARY  ------------------------------------------ #

W = np.array([[1.0, 2.0], [-1.0, 0.5]], np.float32)

B = np.array([0.25, -0.125], np.float32)


def network(path):
    """y = relu(x W^T + b) for x of shape (1, 2), saved to path."""
    nodes = [
        helper.make_node("Gemm", ["x", "W", "b"], ["g"], transB=1),
        helper.make_node("Relu", ["g"], ["y"]),
    ]
    graph = helper.make_graph(
        nodes,
        "g",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 2])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 2])],
        [numpy_helper.from_array(W, "W"), numpy_helper.from_array(B, "b")],
    )
    m = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    m.ir_version = 8
    onnx.save(m, path)


def output(x):
    """The network's output in float32."""
    return np.maximum(np.asarray(x, np.float32) @ W.T + B, 0).astype(np.float32)


def prop_text(lo, hi, threshold):
    """X in [lo, hi]^2 and the unsafe set Y_0 >= threshold."""
    lines = [
        "(declare-const X_0 Real)",
        "(declare-const X_1 Real)",
        "(declare-const Y_0 Real)",
        "(declare-const Y_1 Real)",
    ]
    for i in range(2):
        lines += [f"(assert (<= X_{i} {hi!r}))", f"(assert (>= X_{i} {lo!r}))"]
    return "\n".join(lines + [f"(assert (>= Y_0 {threshold!r}))"]) + "\n"


@contextlib.contextmanager
def official_scoring():
    """The modules of the official scoring, or None without VNNCOMP_SCORING."""
    folder = os.environ.get("VNNCOMP_SCORING")
    if not folder:
        yield None
        return
    sys.path.insert(0, folder)
    try:
        import cachier
        import counterexamples
        import counterexamples_v2

        cachier.disable_caching()  # its cache keys are file names, which the tests reuse
        yield counterexamples, counterexamples_v2
    finally:
        sys.path.remove(folder)

# ----------------------------------------  END OF CODE  ----------------------------------------- #
