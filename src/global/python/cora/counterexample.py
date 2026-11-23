"""counterexample - VNN-COMP counterexamples: written in either format, checked as the scoring does

A counterexample is an input x and the network's output y. write_counterexample gives the text of
VNN-LIB 1.0 ((X_i v) ... (Y_j v)) or 2.0 (per variable a header "X float32 [1, 5]" and one value
per line); every value is printed so that it parses back to the same float32. check_counterexample
replays it as the VNN-COMP 2026 scoring does (vnncomp2026_results/SCORING, counterexamples.py and
counterexamples_v2.py, MIT licensed): the written outputs are ignored, onnxruntime on the CPU
computes them from x cast to the ONNX input type, the input constraints hold exactly (CORRECT) or
up to 1e-4 (CORRECT_UP_TO_TOLERANCE, which the scoring credits but which does not establish sat),
and the output constraints hold with tolerance 0, compared in the scoring's dtypes (1.0: inputs in
float32, the rows of A y in float64; 2.0: every comparison in the variables' float32).

Syntax:   text = write_counterexample(x, y, prop)             prop: a cora.vnnlib.Property
          result, why = check_counterexample("net.onnx", "prop.vnnlib", text)   onnx may be .gz
Outputs:  result - one of CORRECT, CORRECT_UP_TO_TOLERANCE, SPEC_NOT_VIOLATED, NO_CE, MALFORMED_CE,
          WRONG_SHAPE, UNSUPPORTED (the scoring's names)
See also: cora.vnnlib, cora.nn_attack, docs/nn_verification.md

Authors:       Claude Opus 5.5
Written:       21-November-2026
Last update:   23-November-2026
Reviewed:      ---
"""

import functools
import gzip
import math
import re

import numpy as np

from .vnnlib import positional


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

def write_counterexample(x, y, prop):
    """The counterexample text of the input x and output y (flat, C order) for the property's
    VNN-LIB version; values are float32 unless the 2.0 query declares another type."""
    x, y = np.asarray(x).reshape(-1), np.asarray(y).reshape(-1)
    if prop.version == 1:
        lines = [f"(X_{i} {_value_text(v, np.float32)})" for i, v in enumerate(x)]
        lines += [f"(Y_{j} {_value_text(v, np.float32)})" for j, v in enumerate(y)]
        return "(" + "\n".join(lines) + ")\n"
    net, out = prop.query.networks[0], []
    for definitions, values in ((net.inputs, iter(x)), (net.outputs, iter(y))):
        for d in definitions:
            key = d.dtype.name
            dims = ", ".join(str(s) for s in d.shape)
            out.append(f"{d.name} {TYPE_NAMES.get(key, 'real')} [{dims}]")
            out += [
                _value_text(next(values), DTYPES.get(key, np.float64))
                for _ in range(math.prod(d.shape))
            ]
    return "\n".join(out) + "\n"


def check_counterexample(onnx_path, vnnlib_path, content, vnnlib_text=None):
    """(result, why) of the counterexample text for the network and property, as the VNN-COMP 2026
    scoring decides it; the version comes from the property's header (vnnlib_text: read already)."""
    from .vnnlib import VERSION_2, read_text

    text = vnnlib_text if vnnlib_text is not None else read_text(vnnlib_path)
    if not content or len(content.strip()) < 2:
        return NO_CE, "no counterexample"
    return (_check_v2 if VERSION_2.search(text) else _check_v1)(onnx_path, text, content)


# -----------------------------------------  AUXILIARY  ------------------------------------------ #

CORRECT = "correct"

CORRECT_UP_TO_TOLERANCE = "correct_up_to_tolerance"

SPEC_NOT_VIOLATED = "spec_not_violated"

NO_CE = "no_ce"

MALFORMED_CE = "malformed_ce"

WRONG_SHAPE = "wrong_shape"

UNSUPPORTED = "unsupported"

COUNTEREXAMPLE_ATOL = 1e-4  # the scoring's input tolerance (Settings.COUNTEREXAMPLE_ATOL)

ASSIGNMENT_HEADER = re.compile(r"^(\S+)\s+(\S+)\s+\[([0-9,\s]*)\]$")

DTYPES = {
    "F16": np.float16,
    "F32": np.float32,
    "F64": np.float64,
    "Real": np.float64,
    "Unknown": np.float64,
}

TYPE_NAMES = {
    "F16": "float16",
    "F32": "float32",
    "F64": "float64",
    "Real": "real",
    "Unknown": "real",
}

ORT_DTYPES = {
    "tensor(float16)": np.float16,
    "tensor(float)": np.float32,
    "tensor(double)": np.float64,
}


def _value_text(v, dtype):
    """A value as the shortest decimal that parses back to the same number of dtype."""
    return repr(float(dtype(v)))


def _model_bytes(onnx_path):
    """The serialized model of an .onnx or .onnx.gz file."""
    if str(onnx_path).endswith(".gz"):
        with gzip.open(onnx_path, "rb") as stream:
            return stream.read()
    with open(onnx_path, "rb") as stream:
        return stream.read()


@functools.lru_cache(maxsize=2)
def _session_v1(onnx_path):
    """The scoring's 1.0 session: 12 intra and inter threads, no CPU memory arena."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.intra_op_num_threads = 12
    options.inter_op_num_threads = 12
    options.enable_cpu_mem_arena = False
    return ort.InferenceSession(
        _model_bytes(onnx_path), options, providers=["CPUExecutionProvider"]
    )


@functools.lru_cache(maxsize=2)
def _session_v2(onnx_path):
    """The scoring's 2.0 session: the default threads, no CPU memory arena."""
    import onnxruntime as ort

    options = ort.SessionOptions()
    options.enable_cpu_mem_arena = False
    return ort.InferenceSession(
        _model_bytes(onnx_path), options, providers=["CPUExecutionProvider"]
    )


def _io_v1(onnx_path):
    """(input shape, output shape, input dtype) of the graph as the scoring's get_io_nodes."""
    import onnx

    model = onnx.load_model_from_string(_model_bytes(onnx_path))
    sess = _session_v1(onnx_path)
    graph = model.graph
    inp = [n for n in graph.input if n.name == sess.get_inputs()[0].name][0]
    out = [n for n in graph.output if n.name == sess.get_outputs()[0].name][0]
    def shape(v):
        return tuple(d.dim_value if d.dim_value != 0 else 1 for d in v.type.tensor_type.shape.dim)

    dtype = (
        np.float32
        if graph.input[0].type.tensor_type.elem_type == onnx.TensorProto.FLOAT
        else np.float64
    )
    return shape(inp), shape(out), dtype


def _parse_v1(content):
    """The X and Y values of a 1.0 counterexample, as the scoring's get_ce_diff."""
    content = content.replace("\n", " ").strip()
    if len(content) < 2:
        return None
    if content[0] != "(" or content[-1] != ")":
        raise ValueError("not a parenthesized list")
    xs, ys = [], []
    for part in content[1:-1].split(")"):
        part = " ".join(part.split())
        if not part:
            continue
        if part[0] != "(":
            raise ValueError(f"bad entry {part[:40]}")
        name, num = part[1:].split(" ")
        target = xs if name.startswith("X_") else ys if name.startswith("Y_") else None
        if target is None or int(name[2:]) != len(target):
            raise ValueError(f"bad variable {name}")
        target.append(float(num))
    return xs, ys


def _check_v1(onnx_path, vnnlib_text, content):
    """The scoring's 1.0 replay (check_counterexample, get_ce_diff, is_specification_vio)."""
    from .vnnlib import parse_vnnlib

    try:
        parsed = _parse_v1(content)
    except ValueError as error:
        return MALFORMED_CE, str(error)
    if parsed is None:
        return NO_CE, "no counterexample"
    xs, _ = parsed
    in_shape, out_shape, dtype = _io_v1(onnx_path)
    x = np.array(xs, dtype=dtype)
    if x.size != math.prod(in_shape):
        return WRONG_SHAPE, f"cannot reshape input of size {x.size} to {in_shape}"
    sess = _session_v1(onnx_path)
    y = sess.run(None, {sess.get_inputs()[0].name: x.reshape(in_shape)})[0].flatten("C")
    prop = parse_vnnlib(vnnlib_text, math.prod(in_shape), math.prod(out_shape))
    for tol, result in ((0.0, CORRECT), (COUNTEREXAMPLE_ATOL, CORRECT_UP_TO_TOLERANCE)):
        if _violated_v1(prop, xs, tuple(y), dtype, tol):
            return result, f"the replayed output violates the property (input tolerance {tol})"
    return SPEC_NOT_VIOLATED, "the replayed output does not violate the property"


def _violated_v1(prop, xs, y, dtype, tol):
    """is_specification_vio: x (as the input dtype) in a box, A y <= b for a polytope."""
    replay = np.asarray(xs, dtype=dtype).ravel(order="C")
    for case in prop.cases:
        # NumPy 2 compares a float32 element with a Python float in float32, as the scoring does
        if any(
            x < lo - tol or x > hi + tol
            for (lo, hi), x in zip(
                zip(case.lo.tolist(), case.hi.tolist(), strict=True), replay, strict=True
            )
        ):
            continue
        if any(np.all(A.dot(y) <= b + 0.0) for A, b in case.polytopes):
            return True
    return False


def _definitions(query):
    """The variables of a 2.0 query in assignment order: per network inputs, hidden, outputs."""
    return [d for net in query.networks for d in (*net.inputs, *net.hidden, *net.outputs)]


def _parse_v2(content, query):
    """The assignment of a 2.0 counterexample (VNN-LIB 2.0 section 5.3, as the scoring reads it)."""
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    if lines and lines[0] == "sat":
        lines.pop(0)
    assignment, pos = {}, 0
    for d in _definitions(query):
        if pos >= len(lines):
            raise ValueError(f"missing assignment for {d.name}")
        match = ASSIGNMENT_HEADER.fullmatch(lines[pos])
        if not match:
            raise ValueError(f"invalid assignment header {lines[pos]!r}")
        pos += 1
        name, type_name, dims = match.groups()
        shape = [int(v) for v in dims.split(",")] if dims.strip() else []
        key = d.dtype.name
        if name != d.name or shape != list(d.shape):
            raise ValueError(f"expected {d.name} {list(d.shape)}, found {name} {shape}")
        if key not in DTYPES:
            raise NotImplementedError(f"unsupported type {d.dtype}")
        if type_name.lower() not in (TYPE_NAMES[key], "real"):
            raise ValueError(f"{name} has type {type_name}, expected {TYPE_NAMES[key]}")
        count = math.prod(shape)
        if pos + count > len(lines):
            raise ValueError(f"not enough values for {name}")
        assignment[name] = np.asarray(
            [float(v) for v in lines[pos : pos + count]], dtype=DTYPES[key]
        ).reshape(shape)
        pos += count
    if pos != len(lines):
        raise ValueError(f"unexpected content {lines[pos]!r}")
    return assignment


def _check_v2(onnx_path, vnnlib_text, content):
    """The scoring's 2.0 replay (validate_vnnlib2_counterexample) for a query of one network."""
    import vnnlib

    try:
        # exponents written positionally (as VNN-COMP's own 2.0 files), which the parser needs
        query = vnnlib.parse_query_string(positional(vnnlib_text))
        if len(query.networks) != 1 or query.networks[0].hidden:
            return UNSUPPORTED, "one network without hidden variables is supported"
        assignment = _parse_v2(content, query)
    except ValueError as error:
        return MALFORMED_CE, str(error)
    except (NotImplementedError, vnnlib.VNNLibException) as error:
        return UNSUPPORTED, str(error)
    # the inputs cast to the ONNX types, the outputs computed by onnxruntime
    net, sess = query.networks[0], _session_v2(onnx_path)
    feeds = {}
    for d, onnx_in in zip(net.inputs, sess.get_inputs(), strict=True):
        value = assignment[d.name].astype(ORT_DTYPES[onnx_in.type], copy=False)
        assignment[d.name] = value
        shape = onnx_in.shape
        if not (
            len(value.shape) == len(shape)
            and all(
                not isinstance(o, int) or o <= 0 or o == v
                for v, o in zip(value.shape, shape, strict=True)
            )
        ):
            value = value.reshape(shape)
        feeds[onnx_in.name] = value
    outs = sess.run([o.name for o in sess.get_outputs()][: len(net.outputs)], feeds)
    for d, out in zip(net.outputs, outs, strict=True):
        assignment[d.name] = np.asarray(out).reshape(d.shape)
    inputs = {d.name for d in net.inputs}
    for tol, result in ((0.0, CORRECT), (COUNTEREXAMPLE_ATOL, CORRECT_UP_TO_TOLERANCE)):
        if all(
            _eval_bool(
                a.expr,
                assignment,
                tol if _variables(a.expr) and _variables(a.expr) <= inputs else 0.0,
            )
            for a in query.assertions
        ):
            return result, f"the replayed output violates the property (input tolerance {tol})"
    return SPEC_NOT_VIOLATED, "the replayed output does not violate the property"


def _eval_arith(e, assignment):
    """An arithmetic expression on the assignment, in the variables' dtypes."""
    kind = type(e).__name__
    if kind == "Var":
        return assignment[e.name][tuple(e.indices)]
    if kind in ("Float", "Int", "IntExpr"):
        return e.value
    if kind == "Literal":
        return float(e.lexeme)
    if kind == "Negate":
        return -_eval_arith(e.expr, assignment)
    if kind == "Plus":
        return sum(_eval_arith(a, assignment) for a in e.args)
    if kind == "Minus":
        return _eval_arith(e.head, assignment) - sum(_eval_arith(a, assignment) for a in e.rest)
    if kind == "Multiply":
        return math.prod(_eval_arith(a, assignment) for a in e.args)
    raise NotImplementedError(f"unsupported expression {kind}")


def _eval_bool(e, assignment, tol):
    """A Boolean expression on the assignment with the tolerance tol (the scoring's)."""
    kind = type(e).__name__
    if kind == "And":
        return all(_eval_bool(a, assignment, tol) for a in e.args)
    if kind == "Or":
        return any(_eval_bool(a, assignment, tol) for a in e.args)
    lhs, rhs = _eval_arith(e.lhs, assignment), _eval_arith(e.rhs, assignment)
    compare = {
        "GreaterThan": lambda: lhs > rhs - tol,
        "LessThan": lambda: lhs < rhs + tol,
        "GreaterEqual": lambda: lhs >= rhs - tol,
        "LessEqual": lambda: lhs <= rhs + tol,
        "Equal": lambda: abs(lhs - rhs) <= tol,
        "NotEqual": lambda: abs(lhs - rhs) > tol,
    }
    return bool(compare[kind]())


def _variables(e):
    """The names of the variables of an expression."""
    if type(e).__name__ == "Var":
        return {e.name}
    out = set()
    for attr in ("expr", "lhs", "rhs", "head"):
        if hasattr(e, attr):
            out |= _variables(getattr(e, attr))
    for attr in ("args", "rest"):
        for child in getattr(e, attr, ()):
            out |= _variables(child)
    return out

# ----------------------------------------  END OF CODE  ----------------------------------------- #
