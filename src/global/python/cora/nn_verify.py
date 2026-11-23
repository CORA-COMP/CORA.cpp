"""nn_verify - a VNN-COMP instance into the C++ verifier: the network, the property, the CE check

The front end of cora.NeuralNetwork.verify (CORA's verify, in C++: docs/nn_verification.md). An
instance is an ONNX network and a VNN-LIB property, loaded once: the network lowered to the dense
layers of a cora.NeuralNetwork (cora.onnx_network), the property's input boxes and unsafe
polytopes stacked (cora.vnnlib), and the scoring's counterexample check, which C++ calls back for
every candidate (onnxruntime on the float32 point, then VNN-COMP's own check of the written
counterexample). A network that C++ cannot take is "unknown" at once, with the reason, and its
property is not read. The options are cora.NnVerifyOptions; cora.vnncomp_options(category) gives
the preset.

Syntax:   instance = load("net.onnx", "prop.vnnlib")
          result = verify(instance, 60, cora.vnncomp_options("acasxu_2023"))   result.status
See also: cora.onnx_network, cora.vnnlib, cora.counterexample, docs/nn_verification.md

Authors:       Claude Opus 5.5
Written:       21-November-2026
Last update:   23-November-2026
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
    """An instance loaded once: the cora.NeuralNetwork (None, with why, if C++ cannot take the
    graph; then the property is not read), the property and its text, the ONNX file for the
    counterexample check, and the Tensor backend of the network."""

    onnx_path: str
    network: object
    why: str
    prop: object
    vnnlib_text: str
    backend: str = "eigen"


def load(onnx_path, vnnlib_path, backend="eigen", device="cpu"):
    """The instance of the files; the network and the property are read once. backend: the Tensor
    backend of the verifier's network, "eigen" (numpy weights), "torch" (on device) or "auto"
    (choose_backend)."""
    import cora

    # the network first: one the C++ verifier cannot take is unknown without reading the property
    net = OnnxNetwork.load(onnx_path)
    lowered, why = net.layers(drop_output_softmax=True)
    if not lowered:
        return Instance(str(onnx_path), None, why, None, "")
    layers, activations = lowered
    if backend == "auto":
        backend = choose_backend(layers, device)
    convert = (lambda t: t.to(device)) if backend == "torch" else (lambda t: t.numpy())

    def one(layer):
        # a dense pair, or a dict whose tensors move, the residuals' maps as well
        if layer is None:
            return None
        if not isinstance(layer, dict):
            return (convert(layer[0]), convert(layer[1]))
        out = {k: convert(v) if torch.is_tensor(v) else v for k, v in layer.items()}
        if "residuals" in out:
            out["residuals"] = [{**r, "map": one(r["map"])} for r in out["residuals"]]
        return out

    def build(layers, activations):
        # the C++ network, or None and why it refuses a layer
        try:
            return cora.NeuralNetwork([one(layer) for layer in layers], activations), ""
        except (ValueError, TypeError, RuntimeError, NotImplementedError) as error:
            return None, f"the C++ network refuses a layer: {error}"

    network, why = build(layers, activations)
    if network is None:
        return Instance(str(onnx_path), None, why, None, "")
    text = read_text(vnnlib_path)
    prop = parse_vnnlib(text, net.num_inputs, net.num_outputs)
    if net.dropped_softmax and not _compares_outputs(prop):
        # the probabilities themselves: the output Softmax as exp and 1 / sum
        lowered, why = net.layers()
        network, why = build(*lowered) if lowered else (None, why)
        if network is None:
            return Instance(str(onnx_path), None, why, None, "")
    return Instance(str(onnx_path), network, "", prop, text, backend)


def choose_backend(layers, device="cpu"):
    """The faster Tensor backend of a lowered network, as measured on the dev sets: torch for a
    network with a convolution (its GEMMs about twice Eigen's at CORACPP_NATIVE=OFF) or on a GPU,
    eigen for dense networks; eigen if the module has no torch."""
    import cora

    if not cora.has_torch:
        return "eigen"
    if device.startswith("cuda"):
        return "torch"
    conv = any(isinstance(la, dict) and ("conv2d" in la or "conv_transpose2d" in la)
               for la in layers)
    return "torch" if conv else "eigen"


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


def _compares_outputs(prop):
    """Whether every row of the property is y_i - y_j <= 0 like (sum 0, offset 0, or padding)."""
    _, A, b = prop.stacked()
    real = np.isfinite(b)
    return bool(np.all(np.abs(A.sum(-1)[real]) == 0) and np.all(b[real] == 0))


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
