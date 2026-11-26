"""nn_verify - a VNN-COMP instance into the C++ verifier: the network, the property, the CE check

The front end of cora.NeuralNetwork.verify (CORA's verify, in C++: docs/nn_verification.md). An
instance is an ONNX network and a VNN-LIB property, loaded once: the network lowered to the dense
layers of a cora.NeuralNetwork (cora.onnx_network), the property's input boxes and unsafe
polytopes stacked (cora.vnnlib), and the scoring's counterexample check, which C++ calls back for
every candidate (onnxruntime on the float32 point, then VNN-COMP's own check of the written
counterexample). A network that C++ cannot take is "unknown" at once, with the reason, and its
property is not read. The options are cora.NnVerifyOptions; cora.vnncomp_options(category) gives
the preset. A VNN-LIB 2.0 property of several networks (the network field "[('f', 'a.onnx'),
('g', 'b.onnx')]") or of related inputs is verified on one joint network (cora.joint_network), its
counterexample checked on the original networks and property; a 2.0 property that is no union of
output polytopes (products, inputs compared with outputs) by the branch and bound of
cora.nonlinear_property.

Syntax:   instance = load("net.onnx", "prop.vnnlib")
          instance = load("[('f', 'a.onnx'), ('g', 'b.onnx')]", "pair.vnnlib")
          result = verify(instance, 60, cora.vnncomp_options("acasxu_2023"))   result.status
See also: cora.onnx_network, cora.vnnlib, cora.counterexample, cora.joint_network,
          cora.nonlinear_property, docs/nn_verification.md

Authors:       Claude Opus 5.5
Written:       21-November-2026
Last update:   26-November-2026 (several networks, nonlinear properties)
Reviewed:      ---
"""

import time
from dataclasses import dataclass, field

import numpy as np
import torch

from . import counterexample
from .joint_network import InputMap, joint_layers, parse_network_field
from .onnx_network import OnnxNetwork
from .vnnlib import VERSION_2, Case, Property, parse_vnnlib, positional, read_text


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
    parts: list = None  # [(network, case indices)] of a property folded in groups of cases
    merged_relu_pairs: int = 0  # the ReLU pairs folded into linear maps (OnnxNetwork.layers)
    joint: object = None  # the _Joint of a property of several networks or related inputs
    nonlinear: object = None  # (NonlinearProperty, (layers, activations)) beyond output polytopes


def load(onnx_path, vnnlib_path, backend="eigen", device="cpu"):
    """The instance of the files; the network and the property are read once. backend: the Tensor
    backend of the verifier's network, "eigen" (numpy weights), "torch" (on device) or "auto"
    (choose_backend)."""
    import cora

    if len(parse_network_field(onnx_path)) > 1:
        # several networks (VNN-LIB 2.0's equal-to and isomorphic-to pairs): one joint network
        return _load_joint(onnx_path, read_text(vnnlib_path), backend, device)
    # the network first: one the C++ verifier cannot take is unknown without reading the property
    try:
        net = OnnxNetwork.load(onnx_path)
    except (ValueError, NotImplementedError) as error:  # a graph of several inputs
        return Instance(str(onnx_path), None, str(error), None, "")
    lowered, why = net.layers(drop_output_softmax=True)
    groups = [(None, net, lowered)]
    if not lowered and any(r in why for r in FOLDABLE):
        # unless the property fixes inputs whose values make the graph lower (nn4sys' masks), in
        # all its cases at once or in groups of cases
        groups = _folded(net, vnnlib_path) or groups
    if not groups[0][2]:
        return Instance(str(onnx_path), None, why, None, "")
    if backend == "auto":
        backend = choose_backend(groups[0][2][0], device, groups[0][2][1])
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

    text, prop, parts = "", None, []
    for cases, part, lowered in groups:
        network, why = build(*lowered)
        if network is not None and prop is None:
            text = read_text(vnnlib_path)
            try:
                prop = parse_vnnlib(text, net.num_inputs, net.num_outputs)
            except NotImplementedError as error:
                # no union of output polytopes: the branch and bound of the nonlinear property
                return _load_nonlinear(onnx_path, part, text, error)
            if any(c.relations for c in prop.cases):
                # input constraints that relate inputs: the network behind their InputMap
                return _load_joint(str(onnx_path), text, backend, device)
        if network is not None and part.dropped_softmax and not _compares_outputs(prop):
            # the probabilities themselves: the output Softmax as exp and 1 / sum
            lowered, why = part.layers()
            network, why = build(*lowered) if lowered else (None, why)
        if network is None:
            return Instance(str(onnx_path), None, why, None, "")
        parts.append((network, cases))
    return Instance(str(onnx_path), parts[0][0], "", prop, text, backend,
                    parts if len(parts) > 1 else None, getattr(net, "merged_relu_pairs", 0))


def choose_backend(layers, device="cpu", activations=()):
    """The faster Tensor backend of a lowered network, as measured on the dev sets: torch for a
    network with a convolution (its GEMMs about twice Eigen's at CORACPP_NATIVE=OFF), with a
    product (lsnc_relu's graph: 1.7 times the boxes per second) or on a GPU, eigen for the other
    dense networks; eigen if the module has no torch."""
    import cora

    if not cora.has_torch:
        return "eigen"
    if device.startswith("cuda"):
        return "torch"
    conv = any(isinstance(la, dict) and ("conv2d" in la or "conv_transpose2d" in la)
               for la in layers)
    return "torch" if conv or "product" in activations else "eigen"


def verify(instance, timeout, options=None):
    """The verdict on the instance within timeout seconds, by the C++ verifier."""
    import cora

    if instance.network is None:
        return VerifyResult("unknown", stats={"bounding": f"none ({instance.why})"})
    if instance.nonlinear is not None:
        return _verify_nonlinear(instance, timeout)
    options = options or cora.NnVerifyOptions()
    changes = {}
    if instance.parts:
        # a property folded in groups (mscn): many small boxes, each cheap (FOLDED_PARTS), an
        # input generator only for as many inputs as a case leaves free (mscn: 1 to 3 of 308)
        free = max(int(np.count_nonzero(c.hi > c.lo)) for c in instance.prop.cases)
        changes["num_init_gens"] = min(options.num_init_gens, float(max(free, 1)))
        changes["num_approx_err"] = min(options.num_approx_err, FOLDED_PARTS["num_approx_err"])
        changes["mini_batch_size"] = max(options.mini_batch_size, FOLDED_PARTS["mini_batch_size"])
        margin = min(options.verified_margin, FOLDED_PARTS["verified_margin"])
        if margin < options.verified_margin:
            # a thinner margin than the float32 search's buffer: the search in float64
            changes["verified_margin"], changes["dtype"] = margin, "float64"
    before = {name: getattr(options, name) for name in changes}
    for name, value in changes.items():
        setattr(options, name, value)
    try:
        status, stats, x, y = _verify_parts(instance, timeout, options)
    finally:
        for name, value in before.items():
            setattr(options, name, value)
    if instance.merged_relu_pairs:
        stats["merged_relu_pairs"] = instance.merged_relu_pairs
    if status != "sat":
        return VerifyResult(status, stats=stats)
    if instance.joint is not None:
        # the counterexample that the joint confirm check took, in the original inputs
        x, y, text = instance.joint.found
        return VerifyResult("sat", x, y, text, stats)
    x, y = np.asarray(x, dtype=np.float32), np.asarray(y, dtype=np.float32)
    text = counterexample.write_counterexample(x, y, instance.prop)
    return VerifyResult("sat", x, y, text, stats)


# -----------------------------------------  AUXILIARY  ------------------------------------------ #


def _verify_parts(instance, timeout, options):
    """(status, stats, x, y) of C++'s verify on each part of the instance: the parts of fewer
    cases first, each an equal share of the time left (the time an easy part leaves goes to the
    later, larger ones); sat as soon as one part is, unsat only if every part is."""
    confirm = _JointConfirm(instance) if instance.joint is not None else _Confirm(instance)
    parts = sorted(instance.parts or [(instance.network, None)],
                   key=lambda p: 0 if p[1] is None else len(p[1]))
    deadline = time.monotonic() + timeout
    status, stats, note, x, y = "unsat", {}, "", None, None
    for k, (network, cases) in enumerate(parts):
        share = max(0.0, deadline - time.monotonic()) / (len(parts) - k)
        out = _verify_cases(instance.prop, network, cases, share, options, confirm)
        for key, value in out["stats"].items():
            stats[key] = _joined(key, stats.get(key), value)
        note = out["note"] or note
        stats["parts_unsat"] = stats.get("parts_unsat", 0) + (out["status"] == "unsat")
        if out["status"] == "sat":
            status, x, y = "sat", out["x"], out["y"]
            break
        if out["status"] != "unsat":
            status = out["status"]
    stats["bounding"] = f"none ({note})" if note else options.bounding_method
    if len(parts) > 1:
        stats["parts"] = len(parts)
    else:
        stats.pop("parts_unsat")
    return status, stats, x, y


def _joined(key, before, value):
    """A statistic of the parts so far and of one more: the least margin, the largest peak or
    budget or batch, else the sum of counts and seconds."""
    numbers = isinstance(value, (int, float)) and isinstance(before, (int, float))
    if before is None or not numbers:
        return value
    if "margin" in key:
        return min(before, value)
    if any(word in key for word in ("peak", "budget", "batch")):
        return max(before, value)
    return before + value


def _verify_cases(prop, network, cases, timeout, options, confirm):
    """C++'s verify of the network on the property's cases (all of them for None)."""
    if cases is not None:
        prop = Property(prop.version, prop.num_inputs, prop.num_outputs,
                        [prop.cases[k] for k in cases], prop.query)
    case, A, b = prop.stacked()
    lo = np.stack([c.lo for c in prop.cases])
    hi = np.stack([c.hi for c in prop.cases])
    return network.verify(lo, hi, [int(k) for k in case], A, b, timeout, options, confirm)


FOLDABLE = ("of two values", "Div by a value")  # refusals that fixed inputs may lift
# a property folded in groups: input generators for its free inputs only, 10 error slots per
# layer (of up to 768 neurons), batches of up to 4096 boxes; mscn_128d's 3800 cases unsat in 27 s
# against unknown at 50 s with all slots and 1024 (each box 1 to 3 free inputs); pensieve_big (not
# folded) needs its slots and generators: unknown at 59 s with either. mscn's outputs come within
# 4e-5 of its bounds: the margin 1e-5, with the search in float64 (docs/nn_verification.md)
FOLDED_PARTS = {"num_approx_err": 10, "mini_batch_size": 4096, "verified_margin": 1e-5}
MAX_FOLD_GROUPS = 32  # groups of cases folded apart at most (mscn: 10 to 12 masks, 0.7 s each)


def _folded(net, vnnlib_path):
    """The groups [(case indices or None for all, the network with the inputs that the group's
    cases fix to the same values as constants, its lowering)] when each lowers; else None. Each
    network agrees with net on every box of its cases. All cases form one group when that lowers;
    else the cases go in groups by their values of the inputs that make the products linear
    (OnnxNetwork.product_inputs: nn4sys' mscn masks), at most MAX_FOLD_GROUPS."""
    try:
        prop = parse_vnnlib(read_text(vnnlib_path), net.num_inputs, net.num_outputs)
    except (ValueError, NotImplementedError):
        return None
    lo = np.stack([c.lo for c in prop.cases])
    hi = np.stack([c.hi for c in prop.cases])

    def fold(cases):
        # the inputs fixed to one value in all the cases as constants, on the box of the cases
        lo_k, hi_k = lo[cases], hi[cases]
        fixed = (lo_k == hi_k).all(0) & (lo_k == lo_k[0]).all(0)
        if not fixed.any():
            return None, None
        low = np.where(fixed, lo_k[0], lo_k.min(0))
        high = np.where(fixed, hi_k[0], hi_k.max(0))
        folded = net.fold_fixed_inputs(low, high)
        lowered, _ = folded.layers(drop_output_softmax=True)
        return folded, lowered

    every = np.arange(len(prop.cases))
    folded, lowered = fold(every)
    if lowered:
        return [(None, folded, lowered)]
    key = [i for i in net.product_inputs() if (lo[:, i] == hi[:, i]).all()]
    if not key:
        return None
    groups = {}
    for k, values in enumerate(lo[:, key]):
        groups.setdefault(values.tobytes(), []).append(k)
    if not 1 < len(groups) <= MAX_FOLD_GROUPS:
        return None
    out = []
    for cases in groups.values():
        folded, lowered = fold(np.array(cases))
        if not lowered:
            return None
        out.append((cases, folded, lowered))
    return out


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


# Several networks ------------------------------------------------------------------------------

@dataclass
class _Joint:
    """A property of several networks (or of related inputs) as one: the original property and
    its text, the ONNX file of each declared network, the InputMap of the case, the networks in
    float64 (OnnxNetwork) for the check, and the counterexample the confirm check took."""

    prop: object
    text: str
    paths: dict
    imap: object
    nets: list
    found: tuple = None


def _load_joint(field, text, backend, device):
    """The instance of a property of several networks: the joint network (cora.joint_network) on
    the box of its new inputs, the unsafe polytopes those of the stacked outputs. One case (one
    input set) is supported; a network that does not lower is unknown, with the reason."""
    import cora

    from .counterexample import model_paths

    prop = parse_vnnlib(text, None, None)
    if prop.version != 2:
        raise NotImplementedError("nn_verify: several networks need a VNN-LIB 2.0 property")
    if len(prop.cases) != 1:
        raise NotImplementedError("nn_verify: a joint network of one input set is supported")
    paths = model_paths(prop.query, field)
    if any(len(info.inputs) != 1 or len(info.outputs) != 1 for info in prop.networks):
        raise NotImplementedError("nn_verify: a network of one input and one output is supported")
    # each network once per file (equal-to: the same file), lowered without the Softmax drop
    loaded, nets, slices = {}, [], []
    for info in prop.networks:
        path = paths[info.name]
        if path not in loaded:
            net = OnnxNetwork.load(path)
            lowered, why = net.layers()
            loaded[path] = (net, lowered, why)
        net, lowered, why = loaded[path]
        if not lowered:
            return Instance(str(field), None, why, None, "")
        if (net.num_inputs, net.num_outputs) != (info.num_inputs, info.num_outputs):
            raise ValueError(f"nn_verify: the network {info.name} does not match its declaration")
        nets.append(lowered)
        offset = info.inputs[0][2]
        slices.append(slice(offset, offset + info.num_inputs))
    case = prop.cases[0]
    imap = InputMap(case.lo, case.hi, case.relations)
    try:
        layers, activations = joint_layers(imap, nets, slices)
    except NotImplementedError as error:
        return Instance(str(field), None, str(error), None, "")
    if backend == "auto":
        backend = choose_backend(layers, device, activations)
    convert = (lambda a: torch.from_numpy(a).to(device)) if backend == "torch" else (lambda a: a)
    try:
        network = cora.NeuralNetwork([(convert(W), convert(b)) for W, b in layers], activations)
    except (ValueError, TypeError, RuntimeError, NotImplementedError) as error:
        return Instance(str(field), None, f"the C++ network refuses a layer: {error}", None, "")
    # the verifier's property: the box of the new inputs, the polytopes of the stacked outputs
    search = Property(2, imap.dim, prop.num_outputs,
                      [Case(imap.lo, imap.hi, case.polytopes)], prop.query, prop.networks)
    float64 = [loaded[paths[info.name]][0].cast(dtype=torch.float64) for info in prop.networks]
    joint = _Joint(prop, text, paths, imap, float64)
    return Instance(str(field), network, "", search, text, backend, joint=joint)


class _JointConfirm:
    """The check of a candidate w of the joint network: the stacked inputs of w (InputMap) in
    float32, the written counterexample CORRECT by VNN-COMP's checker (onnxruntime on each
    network), and every assertion of the property true for the networks in float64 as well. The
    stacked outputs, or None; the counterexample taken in the instance's joint.found."""

    def __init__(self, instance):
        self.joint = instance.joint

    def __call__(self, w):
        joint = self.joint
        prop = joint.prop
        v = joint.imap.inputs(np.asarray(w, dtype=float))[0].astype(np.float32)
        # each network's outputs by onnxruntime, as the scoring replays them
        ys = []
        for info in prop.networks:
            x = v[info.inputs[0][2] : info.inputs[0][2] + info.num_inputs]
            sess = counterexample._session_v2(joint.paths[info.name])
            onnx_in = sess.get_inputs()[0]
            shape = [d if isinstance(d, int) and d > 0 else 1 for d in onnx_in.shape]
            ys.append(sess.run(None, {onnx_in.name: x.reshape(shape)})[0].reshape(-1))
        y = np.concatenate(ys)
        text = counterexample.write_counterexample(v, y, prop)
        result, _ = counterexample.check_counterexample(joint.paths, None, text, joint.text)
        if result != counterexample.CORRECT or not self._holds_float64(v):
            return None
        joint.found = (v, y.astype(np.float32), text)
        return y.astype(np.float64)

    def _holds_float64(self, v):
        """Whether every assertion on outputs holds on the float32 inputs v with the networks in
        float64 (those of the inputs alone the scoring's check decided, in float32)."""
        joint, assignment, inputs = self.joint, {}, set()
        for info, net in zip(joint.prop.networks, joint.nets, strict=True):
            (name, shape, offset), = info.inputs
            x = torch.from_numpy(v[offset : offset + info.num_inputs].astype(np.float64))
            with torch.no_grad():
                y = net(x.reshape(1, -1)).reshape(-1).numpy()
            assignment[name] = x.numpy().reshape(shape)
            inputs.add(name)
            (out_name, out_shape, _), = info.outputs
            assignment[out_name] = y.reshape(out_shape)
        return all(counterexample._eval_bool(a.expr, assignment, 0.0)
                   for a in joint.prop.query.assertions
                   if not counterexample._variables(a.expr) <= inputs)


# A nonlinear property ----------------------------------------------------------------------------

def _load_nonlinear(onnx_path, net, text, error):
    """The instance of a 2.0 property that is no union of output polytopes (products of
    variables, inputs compared with outputs, !=): its NonlinearProperty on the network in float64
    (Eigen, the output Softmax kept); a property it cannot take raises the parser's error."""
    import vnnlib

    import cora

    from .nonlinear_property import NonlinearProperty

    if not VERSION_2.search(text):
        raise error
    try:
        prop = NonlinearProperty(vnnlib.parse_query_string(positional(text)))
    except NotImplementedError:
        raise error from None
    if (prop.num_inputs, prop.num_outputs) != (net.num_inputs, net.num_outputs):
        raise ValueError("vnnlib: the property's input or output does not match the network")
    lowered, why = net.layers()
    if not lowered or any(isinstance(layer, dict) for layer in lowered[0]):
        return Instance(str(onnx_path), None, why or "nonlinear property: dense layers only",
                        None, "")
    layers = [(W.numpy(), b.numpy()) for W, b in lowered[0]]
    network = cora.NeuralNetwork(layers, lowered[1])
    return Instance(str(onnx_path), network, "", None, text, "eigen",
                    nonlinear=(prop, (layers, list(lowered[1]))))


def _verify_nonlinear(instance, timeout):
    """The branch and bound of a nonlinear property; a counterexample is the scoring's CORRECT
    (onnxruntime's float32 outputs: adaptive_cruise_control's are counterexamples by the float32
    rounding of its clamp, which the real network only touches)."""
    from .nonlinear_property import branch_and_bound

    (prop, layers), onnx_path = instance.nonlinear, instance.onnx_path
    session = counterexample._session_v2(onnx_path)
    onnx_in = session.get_inputs()[0]
    shape = [d if isinstance(d, int) and d > 0 else 1 for d in onnx_in.shape]
    writer = Property(2, prop.num_inputs, prop.num_outputs, [], prop.query)
    found = {}

    def replay(x):
        # the outputs of float32 points by onnxruntime, one point per run (the declared shape)
        return np.stack([session.run(None, {onnx_in.name: p.reshape(shape)})[0].reshape(-1)
                         for p in x]) if len(x) else np.zeros((0, prop.num_outputs), np.float32)

    def check(x):
        y = replay(x[None])[0]
        text = counterexample.write_counterexample(x, y, writer)
        result, _ = counterexample.check_counterexample(onnx_path, None, text, instance.vnnlib_text)
        if result != counterexample.CORRECT:
            return False
        found.update(x=x, y=y.astype(np.float32), text=text)
        return True

    start = time.monotonic()
    status, _, stats = branch_and_bound(prop, instance.network, layers, timeout, replay, check)
    stats = {f"nonlinear_{k}": v for k, v in stats.items()}
    stats.update(bounding="nonlinear-bab", seconds=time.monotonic() - start)
    if status != "sat":
        return VerifyResult(status, stats=stats)
    return VerifyResult("sat", found["x"], found["y"], found["text"], stats)

# ----------------------------------------  END OF CODE  ----------------------------------------- #
