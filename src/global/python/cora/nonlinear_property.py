"""nonlinear_property - a VNN-LIB 2.0 property beyond polytopes: branch and bound on the input box

A property whose assertions multiply variables or compare inputs with outputs (VNN-COMP 2026's
adaptive_cruise_control_non_linear: (>= (* X[0,0] 200.0) (* X[0,1] X[0,1])), (!= Y[0,0] 0.0)) is no
union of output polytopes. It is decided on boxes of the inputs instead: the network's outputs on
a box are bounded by zonotopes (cora.NeuralNetwork.evaluate_zonotope_batch), every assertion is
evaluated on the intervals of the inputs and the outputs in three-valued logic (Kleene's and, or:
can it hold, can it fail), and a box on which some assertion cannot hold is safe. The other
boxes are halved along their widest input (relative to the property's box) until all are safe
(unsat) or the time is over (unknown); the centers and random points of the open boxes are tried
on the way, each as the scoring's check of a written counterexample decides (sat).

The scoring replays a counterexample with onnxruntime in float32, whose rounding can cross a
threshold that the real network only touches (adaptive_cruise_control's clamp to 100.001: its
float32 output reaches 100.0011 where the real one is 100.001). The output bounds are therefore
widened by a bound of the float32 error (float32_error, a forward error analysis of the affine
layers); the assertions are evaluated as the scoring does, in float32: constants as their float32
values, every operation widened by a relative 2^-23, and an output compared with constants split
into cells of float32 values (the constants themselves and the values between them).

Syntax:   p = NonlinearProperty(query)               query: vnnlib.parse_query_string(text)
          p.lo, p.hi                                 the input box of the property
          holds = p.can_hold(xlo, xhi, ylo, yhi)     (B,) bool: some point of the boxes may hold
          ok = p.holds(x, y)                         (B,) bool: the points meet every assertion
          status, x, stats = branch_and_bound(p, net, (layers, acts), timeout, replay, check)
See also: cora.vnnlib, cora.nn_verify, cora.counterexample

Authors:       Claude Opus 5.5
Written:       25-November-2026
Last update:   ---
Reviewed:      ---
"""

import math
import time

import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

class NonlinearProperty:
    """The assertions of a 2.0 query of one network with one input and one output, the box of its
    input from the bounds of single inputs (finite, else ValueError)."""

    def __init__(self, query):
        if len(query.networks) != 1 or query.networks[0].hidden:
            raise NotImplementedError("nonlinear_property: one network without hidden variables")
        net = query.networks[0]
        if len(net.inputs) != 1 or len(net.outputs) != 1:
            raise NotImplementedError("nonlinear_property: one input and one output")
        self.query = query
        self.input, self.output = net.inputs[0], net.outputs[0]
        self.input_shape = tuple(int(s) for s in self.input.shape)
        self.output_shape = tuple(int(s) for s in self.output.shape)
        self.num_inputs = math.prod(self.input_shape)
        self.num_outputs = math.prod(self.output_shape)
        self.assertions = [a.expr for a in query.assertions]
        self.lo, self.hi = np.full(self.num_inputs, -np.inf), np.full(self.num_inputs, np.inf)
        for e in self.assertions:
            for atom in _top_conjuncts(e):
                self._bound(atom)
        # the constants that a variable element is compared with alone (float32 values)
        self.thresholds = {}
        for e in self.assertions:
            self._collect(e)
        if not (np.isfinite(self.lo).all() and np.isfinite(self.hi).all()):
            raise ValueError("vnnlib: an input is unbounded")
        if np.any(self.lo > self.hi):
            raise ValueError("vnnlib: the range of an input is empty")

    def can_hold(self, xlo, xhi, ylo, yhi):
        """Per box (rows of xlo, xhi (B, n) and of the output bounds ylo, yhi (B, m)) whether every
        assertion may hold on it (False: the box has no counterexample). A variable compared with
        constants alone (Y == 0.0, Y < 100.0, (- X[0,1]) >= 0.0) is split into its cells: each
        constant itself and the float32 values between two of them; the box may hold if some
        product of cells may (then Y == 0 and Y != 0 never both may hold)."""
        bounds = {self.input.name: [np.array(xlo, float), np.array(xhi, float)],
                  self.output.name: [np.array(ylo, float), np.array(yhi, float)]}
        out = np.zeros(bounds[self.input.name][0].shape[0], dtype=bool)
        for cell in self._cells():
            alive = ~out
            env = {}
            for name, (lo, hi) in bounds.items():
                lo, hi = lo.copy(), hi.copy()
                for (var, flat), (low, high) in cell:
                    if var == name:
                        lo[:, flat], hi[:, flat] = np.maximum(lo[:, flat], low), \
                            np.minimum(hi[:, flat], high)
                alive &= np.all(lo <= hi, axis=1)
                env[name] = (lo, hi)
            if not alive.any():
                continue
            env = {name: (lo[alive], hi[alive]) for name, (lo, hi) in env.items()}
            holds = np.ones(int(alive.sum()), dtype=bool)
            for e in self.assertions:
                holds &= _three_valued(e, env)[0]
            out[np.flatnonzero(alive)[holds]] = True
        return out

    def _cells(self):
        """The products of the cells of the variables compared with constants alone, at most
        MAX_CELLS (the variables of the most constants first): each a list of ((name, flat),
        (low, high)) of float32 values."""
        own = {}
        for key, values in self.thresholds.items():
            own[key] = _cells_of(sorted(values))
            if key[0] == self.input.name:
                # an input's cells within the property's box
                lo, hi = self.lo[key[1]], self.hi[key[1]]
                own[key] = [(a, b) for a, b in own[key] if a <= hi and b >= lo]
        cells = [[]]
        for key in sorted(own, key=lambda k: -len(own[k])):
            if len(own[key]) > 1 and len(cells) * len(own[key]) <= MAX_CELLS:
                cells = [c + [(key, o)] for c in cells for o in own[key]]
        return cells

    def holds(self, x, y, outputs_only=False):
        """Per point (rows of x (B, n) and y (B, m)) whether every assertion holds (outputs_only:
        every one that reads an output), compared in the dtype of x and y (float32: as the
        scoring's check)."""
        env = {self.input.name: np.asarray(x), self.output.name: np.asarray(y)}
        out = np.ones(env[self.input.name].shape[0], dtype=bool)
        for e in self.assertions:
            if outputs_only and self.output.name not in _names(e):
                continue
            out &= _pointwise(e, env)
        return out

    def _collect(self, e):
        """The constants that a variable is compared with alone (a linear comparison of one
        variable element) in e, as float32 values."""
        kind = type(e).__name__
        if kind in ("And", "Or"):
            for a in e.args:
                self._collect(a)
            return
        try:
            lhs, rhs = e.lhs.to_linear_expr(), e.rhs.to_linear_expr()
        except Exception:  # noqa: BLE001  a product of variables: no threshold
            return
        coeffs = {}
        for sign, expr in ((1.0, lhs), (-1.0, rhs)):
            for term in expr.terms:
                key = (term.var.name, _flat(term.var))
                coeffs[key] = coeffs.get(key, 0.0) + sign * term.coeff
        coeffs = {k: a for k, a in coeffs.items() if a != 0}
        if len(coeffs) == 1:
            ((key, a),) = coeffs.items()
            value = -(lhs.constant - rhs.constant) / a
            self.thresholds.setdefault(key, set()).add(float(np.float32(value)))

    def _bound(self, atom):
        """A bound of one input from a comparison of an input element with a constant."""
        kind = type(atom).__name__
        if kind not in ("LessEqual", "LessThan", "GreaterEqual", "GreaterThan", "Equal"):
            return
        lhs, rhs = atom.lhs, atom.rhs
        flip = False
        if type(lhs).__name__ != "Var":
            lhs, rhs, flip = rhs, lhs, True
        if type(lhs).__name__ != "Var" or lhs.name != self.input.name:
            return
        value = _constant(rhs)
        if value is None:
            return
        i = int(np.ravel_multi_index(tuple(lhs.indices), self.input_shape))
        upper = kind in ("LessEqual", "LessThan")
        if flip and kind != "Equal":
            upper = not upper
        # the bound and its float32 value, which the scoring compares a float32 input with
        value32 = float(np.float32(value))
        if kind == "Equal" or upper:
            self.hi[i] = min(self.hi[i], max(value, value32))
        if kind == "Equal" or not upper:
            self.lo[i] = max(self.lo[i], min(value, value32))


def branch_and_bound(prop, net, layers, timeout, replay, check, batch=2048, samples=128,
                     max_boxes=2**22, seed=0):
    """(status, x, stats) of the property on the network within timeout seconds. net: the
    cora.NeuralNetwork in float64 of the bounds; layers: its ([(W, b)], activations) for the
    float32 error (float32_error), by which the output bounds widen. "unsat" when every box is
    safe, "sat" with the float32 input x that check(x) accepts (the scoring's check of the written
    counterexample), else "unknown". Per step, batch boxes; the centers, corners (and those one
    float32 step inside) and random points of samples open boxes are replayed (replay(x): the
    float32 outputs (B, m) as the scoring computes them) and those that meet every assertion go to
    check; at most max_boxes open boxes."""
    deadline = time.monotonic() + timeout
    rng = np.random.default_rng(seed)
    width = np.where(prop.hi > prop.lo, prop.hi - prop.lo, 1.0)
    lo, hi = prop.lo[None, :], prop.hi[None, :]
    stats = {"boxes": 0, "points": 0, "candidates": 0}

    def try_points(x):
        # the float32 points in the box, replayed; those that meet every assertion to the check
        x = np.unique(x.astype(np.float32), axis=0)
        x = x[np.all((x >= prop.lo) & (x <= prop.hi), axis=1)]
        stats["points"] += x.shape[0]
        for k in np.flatnonzero(prop.holds(x, replay(x))):
            stats["candidates"] += 1
            if check(x[k]):
                return x[k]
        return None

    while lo.shape[0]:
        if time.monotonic() > deadline or lo.shape[0] > max_boxes:
            return "unknown", None, stats
        # the next boxes, their output bounds widened by the float32 error, the ones that may hold
        blo, bhi, lo, hi = lo[:batch], hi[:batch], lo[batch:], hi[batch:]
        ylo, yhi = output_bounds(net, blo, bhi)
        err = float32_error(*layers, blo, bhi)
        stats["boxes"] += blo.shape[0]
        keep = prop.can_hold(blo, bhi, ylo - err, yhi + err)
        blo, bhi = blo[keep], bhi[keep]
        if not blo.shape[0]:
            continue
        # their centers and corners, a few random points, then the halves along the widest input
        pick = rng.permutation(blo.shape[0])[:samples]
        centers = (blo[pick] + bhi[pick]) / 2
        corners = np.where(rng.random((len(pick), blo.shape[1])) < 0.5, blo[pick], bhi[pick])
        # a corner one float32 step inside as well (a strict bound on the box's face: X > 0)
        nudged = np.nextafter(corners.astype(np.float32), centers.astype(np.float32))
        inside = blo[pick] + rng.random((len(pick), blo.shape[1])) * (bhi[pick] - blo[pick])
        found = try_points(np.vstack([centers, corners, nudged, inside]))
        if found is not None:
            return "sat", found, stats
        dim = np.argmax((bhi - blo) / width, axis=1)
        rows = np.arange(blo.shape[0])
        mid = (blo[rows, dim] + bhi[rows, dim]) / 2
        left_hi, right_lo = bhi.copy(), blo.copy()
        left_hi[rows, dim], right_lo[rows, dim] = mid, mid
        lo = np.vstack([lo, blo, right_lo])
        hi = np.vstack([hi, left_hi, bhi])
    return "unsat", None, stats


def float32_error(layers, activations, lo, hi):
    """A bound (B, m) of how far a float32 evaluation of the network (onnxruntime, as the scoring
    replays a counterexample) may lie from the real outputs on the boxes [lo, hi] (B, n) of float32
    inputs. The rounding of an affine layer z_k = W_k h + b_k of k terms is at most
    r_k = gamma_{k+2} (|W_k| |h| + |b_k|) (any order of summation; twice that, for the affine maps
    that the lowering merged), |h| by intervals; it reaches the output through the Jacobian
    d y / d z_k, bounded by interval matrices over the boxes (a ReLU's slope 1 if active, 0 if
    inactive, [0, 1] if neither, its input widened by the errors so far). layers: [(W, b)] as
    numpy arrays; activations: "relu" or "identity" (another: an infinite error)."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    if any(act not in ("relu", "identity") for act in activations):
        return np.full((lo.shape[0], np.asarray(layers[-1][0]).shape[0]), np.inf)
    # forward: the rounding of each layer and the slopes of its activation
    rounding, slopes, carried = [], [], np.zeros_like(lo)
    for (W, b), act in zip(layers, activations, strict=True):
        W, b = np.asarray(W, float), np.asarray(b, float)
        k = W.shape[1]
        gamma = (k + 2) * UNIT32 / (1 - (k + 2) * UNIT32)
        mag = np.maximum(np.abs(lo), np.abs(hi)) + carried
        r = 2 * gamma * (mag @ np.abs(W).T + np.abs(b))
        center, radius = (lo + hi) / 2, (hi - lo) / 2
        zc, zr = center @ W.T + b, radius @ np.abs(W).T
        # the errors so far through |W| (an upper bound, for the ReLUs' states only)
        carried = carried @ np.abs(W).T + r
        lo, hi = zc - zr, zc + zr
        if act == "relu":
            slopes.append((np.where(lo - carried > 0, 1.0, 0.0),
                           np.where(hi + carried > 0, 1.0, 0.0)))
            lo, hi = np.maximum(lo, 0.0), np.maximum(hi, 0.0)
        else:
            slopes.append((np.ones_like(lo), np.ones_like(lo)))
        rounding.append(r)
    # backward: the interval Jacobian of the output by each layer's z, times its rounding
    B, m = lo.shape
    dlo, dhi = slopes[-1]
    Mlo = np.eye(m)[None] * dlo[:, None, :]
    Mhi = np.eye(m)[None] * dhi[:, None, :]
    error = np.einsum("bij,bj->bi", np.maximum(np.abs(Mlo), np.abs(Mhi)), rounding[-1])
    for k in range(len(layers) - 2, -1, -1):
        W = np.abs(np.asarray(layers[k + 1][0], float)), np.asarray(layers[k + 1][0], float)
        center, radius = (Mlo + Mhi) / 2 @ W[1], (Mhi - Mlo) / 2 @ W[0]
        alo, ahi = center - radius, center + radius
        dlo, dhi = (d[:, None, :] for d in slopes[k])
        Mlo = np.minimum(alo * dlo, alo * dhi)
        Mhi = np.maximum(ahi * dlo, ahi * dhi)
        error += np.einsum("bij,bj->bi", np.maximum(np.abs(Mlo), np.abs(Mhi)), rounding[k])
    return error


def output_bounds(net, lo, hi):
    """The bounds (B, m) of the network's outputs on the boxes [lo, hi] (B, n): the interval hull
    of the output zonotopes of evaluate_zonotope_batch, widened by RELATIVE."""
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    c = (lo + hi) / 2
    r = (hi - lo) / 2
    G = np.zeros((lo.shape[0], lo.shape[1], lo.shape[1]))
    idx = np.arange(lo.shape[1])
    G[:, idx, idx] = r
    yc, yG, dr = net.evaluate_zonotope_batch(c, G)
    yc, yG = np.asarray(yc), np.asarray(yG)
    rad = np.abs(yG).sum(axis=1)
    if dr is not None:
        rad = rad + np.asarray(dr)
    rad = rad + RELATIVE * (np.abs(yc) + rad)
    return yc - rad, yc + rad


# -----------------------------------------  AUXILIARY  ------------------------------------------ #

RELATIVE = 2.0**-23  # the widening of an interval per operation (float32's rounding, outward)

UNIT32 = 2.0**-24  # the unit roundoff of float32

TINY32 = 2.0**-149  # the least subnormal float32

MAX_CELLS = 256  # products of cells of the variables compared with constants, at most


def _top_conjuncts(e):
    """The atoms of the top-level conjunction of an expression."""
    if type(e).__name__ == "And":
        return [a for arg in e.args for a in _top_conjuncts(arg)]
    return [e]


def _constant(e):
    """The value of a constant expression (a number or its negation), else None."""
    kind = type(e).__name__
    if kind in ("Float", "Int", "IntExpr"):
        return float(e.value)
    if kind == "Literal":
        return float(e.lexeme)
    if kind == "Negate":
        value = _constant(e.expr)
        return None if value is None else -value
    return None


def _widen(lo, hi):
    """An interval widened outward by RELATIVE of each end's magnitude and the least subnormal
    float32 (an operation's float32 rounding is within half an ulp of its exact result, and
    rounding is monotone)."""
    return lo - RELATIVE * np.abs(lo) - TINY32, hi + RELATIVE * np.abs(hi) + TINY32


def _interval(e, env):
    """The interval (lo, hi) of an arithmetic expression on the boxes env {name: (lo, hi)}."""
    kind = type(e).__name__
    if kind == "Var":
        lo, hi = env[e.name]
        flat = _flat(e)
        return lo[:, flat], hi[:, flat]
    value = _constant(e)
    if value is not None:
        # a constant as the scoring compares it with float32 values: its float32 value
        value = float(np.float32(value))
        return value, value
    if kind == "Negate":
        lo, hi = _interval(e.expr, env)
        return -hi, -lo
    if kind == "Plus":
        parts = [_interval(a, env) for a in e.args]
        return _widen(sum(p[0] for p in parts), sum(p[1] for p in parts))
    if kind == "Minus":
        lo, hi = _interval(e.head, env)
        for a in e.rest:
            alo, ahi = _interval(a, env)
            lo, hi = lo - ahi, hi - alo
        return _widen(lo, hi)
    if kind == "Multiply":
        args = list(e.args)
        lo, hi = _interval(args[0], env)
        for k in range(1, len(args)):
            alo, ahi = _interval(args[k], env)
            if len(args) == 2 and _same_var(args[0], args[1]):
                # a square: not below 0
                sq = np.maximum(lo * lo, hi * hi)
                low = np.where((lo <= 0) & (hi >= 0), 0.0, np.minimum(lo * lo, hi * hi))
                lo, hi = low, sq
            else:
                products = [lo * alo, lo * ahi, hi * alo, hi * ahi]
                lo, hi = np.minimum.reduce(products), np.maximum.reduce(products)
            lo, hi = _widen(lo, hi)
        return lo, hi
    raise NotImplementedError(f"nonlinear_property: the expression {kind} is not supported")


def _cells_of(values):
    """The cells of a variable compared with the sorted float32 values: each value, and the
    float32 values strictly between two neighbors (and below the first, above the last)."""
    cells = [(v, v) for v in values]
    edges = [-np.inf, *values, np.inf]
    up, down = np.float32(np.inf), np.float32(-np.inf)
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        low = -np.inf if a == -np.inf else float(np.nextafter(np.float32(a), up))
        high = np.inf if b == np.inf else float(np.nextafter(np.float32(b), down))
        if low <= high:
            cells.append((low, high))
    return cells


def _names(e):
    """The names of the variables of an expression."""
    if type(e).__name__ == "Var":
        return {e.name}
    out = set()
    for attr in ("expr", "lhs", "rhs", "head"):
        if hasattr(e, attr):
            out |= _names(getattr(e, attr))
    for attr in ("args", "rest"):
        for child in getattr(e, attr, ()):
            out |= _names(child)
    return out


def _flat(var):
    """The flat (C order) index of a variable's element."""
    return int(np.ravel_multi_index(tuple(var.indices), tuple(int(s) for s in var.shape)))


def _same_var(a, b):
    """Whether two expressions are the same element of a variable (a product of it a square)."""
    return (type(a).__name__ == type(b).__name__ == "Var" and a.name == b.name
            and list(a.indices) == list(b.indices))


def _three_valued(e, env):
    """(may hold, may fail) per box of a Boolean expression: Kleene's and, or over comparisons of
    intervals."""
    kind = type(e).__name__
    if kind == "And":
        parts = [_three_valued(a, env) for a in e.args]
        return np.logical_and.reduce([p[0] for p in parts]), \
            np.logical_or.reduce([p[1] for p in parts])
    if kind == "Or":
        parts = [_three_valued(a, env) for a in e.args]
        return np.logical_or.reduce([p[0] for p in parts]), \
            np.logical_and.reduce([p[1] for p in parts])
    n = next(iter(env.values()))[0].shape[0]
    alo, ahi = _interval(e.lhs, env)
    blo, bhi = _interval(e.rhs, env)
    alo, ahi, blo, bhi = (np.broadcast_to(np.asarray(v, float), (n,)) for v in (alo, ahi, blo, bhi))
    point = (alo == ahi) & (blo == bhi) & (alo == blo)
    overlap = (alo <= bhi) & (blo <= ahi)
    if kind == "LessThan":
        return alo < bhi, ahi >= blo
    if kind == "LessEqual":
        return alo <= bhi, ahi > blo
    if kind == "GreaterThan":
        return ahi > blo, alo <= bhi
    if kind == "GreaterEqual":
        return ahi >= blo, alo < bhi
    if kind == "Equal":
        return overlap, ~point
    if kind == "NotEqual":
        return ~point, overlap
    raise NotImplementedError(f"nonlinear_property: the expression {kind} is not supported")


def _value(e, env):
    """An arithmetic expression per point, in the dtype of the variables (as the scoring's)."""
    kind = type(e).__name__
    if kind == "Var":
        return env[e.name][:, _flat(e)]
    value = _constant(e)
    if value is not None and kind != "Negate":
        return value
    if kind == "Negate":
        return -_value(e.expr, env)
    if kind == "Plus":
        return sum(_value(a, env) for a in e.args)
    if kind == "Minus":
        return _value(e.head, env) - sum(_value(a, env) for a in e.rest)
    if kind == "Multiply":
        return math.prod(_value(a, env) for a in e.args)
    raise NotImplementedError(f"nonlinear_property: the expression {kind} is not supported")


def _pointwise(e, env):
    """A Boolean expression per point (B,)."""
    kind = type(e).__name__
    if kind == "And":
        return np.logical_and.reduce([_pointwise(a, env) for a in e.args])
    if kind == "Or":
        return np.logical_or.reduce([_pointwise(a, env) for a in e.args])
    a, b = _value(e.lhs, env), _value(e.rhs, env)
    compare = {"LessThan": np.less, "LessEqual": np.less_equal, "GreaterThan": np.greater,
               "GreaterEqual": np.greater_equal, "Equal": np.equal, "NotEqual": np.not_equal}
    if kind not in compare:
        raise NotImplementedError(f"nonlinear_property: the expression {kind} is not supported")
    n = next(iter(env.values())).shape[0]
    return np.broadcast_to(compare[kind](a, b), (n,))

# ----------------------------------------  END OF CODE  ----------------------------------------- #
