"""nn_verify - a VNN-COMP instance into the C++ verifier: the network, the property, the CE check

The front end of cora.NeuralNetwork.verify (CORA's verify, in C++: docs/nn_verification.md). An
instance is an ONNX network and a VNN-LIB property, loaded once: the network lowered to the dense
layers of a cora.NeuralNetwork (cora.onnx_network), the property's input boxes and unsafe
polytopes stacked (cora.vnnlib), and the scoring's counterexample check, which C++ calls back for
every candidate (onnxruntime on the float32 point, then VNN-COMP's own check of the written
counterexample). A network that is no chain of affine maps and activations is "unknown" with the
reason. The options are cora.NnVerifyOptions; cora.vnncomp_options(category) gives the preset.

Syntax:   instance = load("net.onnx", "prop.vnnlib")
          result = verify(instance, 60, cora.vnncomp_options("acasxu_2023"))   result.status
See also: cora.onnx_network, cora.vnnlib, cora.counterexample, docs/nn_verification.md

Authors:       Claude Opus 5.5
Written:       21-November-2026
Last update:   ---
Reviewed:      ---
"""

from dataclasses import dataclass, field

import numpy as np
import torch

from . import counterexample
from .onnx_network import OnnxNetwork
from .vnnlib import parse_vnnlib, read_text


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #


@dataclass
class VerifyResult:
    """The verdict and, for sat, the counterexample (rows x, y) with its text; stats for the CSV."""

    status: str
    x: object = None
    y: object = None
    counterexample: str = ""
    stats: dict = field(default_factory=dict)


@dataclass
class Instance:
    """An instance loaded once: the cora.NeuralNetwork (None, with why, if the graph is no dense
    chain), the property and its text, the ONNX file for the counterexample check."""

    onnx_path: str
    network: object
    why: str
    prop: object
    vnnlib_text: str


def load(onnx_path, vnnlib_path, backend="eigen", device="cpu"):
    """The instance of the files; the network and the property are read once. backend: the Tensor
    backend of the verifier's network, "eigen" (numpy weights) or "torch" (on device)."""
    import cora

    net = OnnxNetwork.load(onnx_path)
    text = read_text(vnnlib_path)
    prop = parse_vnnlib(text, net.num_inputs, net.num_outputs)
    lowered, why = net.layers()
    network = None
    if lowered:
        layers, activations = lowered
        convert = (lambda t: t.to(device)) if backend == "torch" else (lambda t: t.numpy())

        def one(layer):
            # a dense pair, or a conv2d or gather dict whose tensors move
            if not isinstance(layer, dict):
                return (convert(layer[0]), convert(layer[1]))
            return {k: convert(v) if torch.is_tensor(v) else v for k, v in layer.items()}

        network = cora.NeuralNetwork([one(layer) for layer in layers], activations)
    return Instance(str(onnx_path), network, why or "", prop, text)


def verify(instance, timeout, options=None):
    """The verdict on the instance within timeout seconds, by the C++ verifier."""
    import cora

    if instance.network is None:
        return VerifyResult("unknown", stats={"bounding": f"none ({instance.why})"})
    options = options or cora.NnVerifyOptions()
    case, A, b = instance.prop.stacked()
    lo = np.stack([c.lo for c in instance.prop.cases])
    hi = np.stack([c.hi for c in instance.prop.cases])
    confirm = _Confirm(instance)
    out = instance.network.verify(lo, hi, [int(k) for k in case], A, b, timeout, options, confirm)
    stats = dict(out["stats"], bounding=options.bounding_method)
    if out["note"]:
        stats["bounding"] = f"none ({out['note']})"
    if out["status"] != "sat":
        return VerifyResult(out["status"], stats=stats)
    x, y = np.asarray(out["x"], dtype=np.float32), np.asarray(out["y"], dtype=np.float32)
    text = counterexample.write_counterexample(x, y, instance.prop)
    return VerifyResult("sat", x, y, text, stats)


# -----------------------------------------  AUXILIARY  ------------------------------------------ #


class _Confirm:
    """The scoring's check of a candidate: onnxruntime's output of the float32 point, and the
    written counterexample CORRECT by VNN-COMP's checker; the output, or None."""

    def __init__(self, instance):
        self.instance = instance
        self.session = counterexample._session_v1(instance.onnx_path)
        shape = self.session.get_inputs()[0].shape
        self.shape = [d if isinstance(d, int) and d > 0 else 1 for d in shape]
        self.name = self.session.get_inputs()[0].name

    def __call__(self, x):
        x = np.asarray(x, dtype=np.float32)
        y = self.session.run(None, {self.name: x.reshape(self.shape)})[0].reshape(-1)
        text = counterexample.write_counterexample(x, y, self.instance.prop)
        result, _ = counterexample.check_counterexample(
            self.instance.onnx_path, None, text, self.instance.vnnlib_text
        )
        return y.astype(np.float64) if result == counterexample.CORRECT else None


# ----------------------------------------  END OF CODE  ----------------------------------------- #
