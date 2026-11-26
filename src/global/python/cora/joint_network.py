"""joint_network - a property of several networks as one network on a box: VNN-LIB 2.0's pairs

VNN-LIB 2.0 declares several networks in one query (VNN-COMP 2026: f and g, equal-to or
isomorphic-to) whose inputs are related: X_g[i] == X_f[i], X_g[0] <= X_f[0]. The verifier takes
one network and an input box, so the property becomes one: the joint network reads new inputs w
from a box, maps them to the stacked inputs v = [X_f; X_g] (InputMap) and runs the networks side by
side (block-diagonal layers), its output the stacked outputs [Y_f; Y_g]; the unsafe polytopes of
the property are those of the stacked outputs already.

InputMap from the boxes and the relations sum_i a_i v_i + c <= 0 of a case:
1. an equality of two inputs (a pair of opposite relations) makes one an affine image of the other
   (v_j = s v_i + t); the representatives z are the free inputs, v = S z + t exactly;
2. the boxes of the inputs bound the representatives, exactly, tightened by the inequalities (bound
   propagation);
3. each independent inequality g z <= h is a new input w_k = g z, bounded by the box of z and h; the
   rest of w are representatives (unit rows); z = M^-1 w;
4. a representative that is no input of w (one that an inequality determines) may leave its box
   on the box of w: it is clamped into its box by a first ReLU layer (lo + relu(e - lo) -
   relu(e - hi)), which leaves the other inputs as they are (relu(w - lo(w)) + lo(w)).
The box of w maps onto a superset of the feasible inputs (unsat is sound); every point of it maps
to inputs that meet the boxes, the equalities and the inequalities of step 3 (exactly for the
structures of the benchmarks: a bound of a representative that a clamp restores cannot break an
inequality whose input it is). Inequalities beyond the dimension are not enforced: feasible()
tells, and a counterexample is always checked against the original property.

Syntax:   imap = InputMap(lo, hi, relations)          relations: [({input: a}, c)] of a case
          imap.lo, imap.hi                            the box of the joint network's inputs w
          v = imap.inputs(w)                          the stacked inputs of rows w (float64)
          layers, activations = joint_layers(imap, [(layers, acts), ...], [slice, ...])
          pairs = parse_network_field("[('f', 'a.onnx'), ('g', 'b.onnx')]")
See also: cora.vnnlib, cora.nn_verify, cora.counterexample

Authors:       Claude Opus 5.5
Written:       25-November-2026
Last update:   ---
Reviewed:      ---
"""

import ast

import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

def parse_network_field(field):
    """The (name, path) pairs of a network field of VNN-COMP 2026's instances.csv: a Python literal
    list of pairs "[('f', 'a.onnx'), ('g', 'b.onnx')]", or one path (name None)."""
    text = str(field).strip()
    if not text.startswith("["):
        return [(None, text)]
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError) as error:
        raise ValueError(f"joint_network: the network field is no list of pairs: {error}") from None
    if not isinstance(parsed, list) or not parsed:
        raise ValueError("joint_network: the network field is no list of pairs")
    pairs = []
    for entry in parsed:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError(f"joint_network: a network entry is no pair: {entry!r}")
        pairs.append((str(entry[0]), str(entry[1])))
    return pairs


class InputMap:
    """The box of new inputs w and the map from it onto the stacked inputs v of a case (the module
    doc): lo, hi (k,), the affine part v = A h + c of the front's output h, and the ReLU layer
    (W0, b0) of h = relu(W0 w + b0) when a representative is clamped (else h = w)."""

    def __init__(self, lo, hi, relations, tol=None):
        from cora import default_tol

        # the agreement of equalities that close a cycle (None: the float64 default tolerance)
        tol = default_tol("float64") if tol is None else tol
        lo, hi = np.asarray(lo, dtype=float), np.asarray(hi, dtype=float)
        n = lo.shape[0]
        self.n = n
        equalities, inequalities = _split_relations(relations)
        # 1. the equalities of two inputs: each input an affine image of its representative
        root, scale, shift = _union(n, [e for e in equalities if len(e[0]) == 2], tol)
        inequalities += [r for e in equalities if len(e[0]) != 2 for r in _both(e)]
        reps = sorted(set(root))
        index = {r: k for k, r in enumerate(reps)}
        K = len(reps)
        S = np.zeros((n, K))
        for i in range(n):
            S[i, index[root[i]]] = scale[i]
        t = np.array(shift)
        # 2. the boxes of the inputs as boxes of the representatives
        zlo, zhi = np.full(K, -np.inf), np.full(K, np.inf)
        for i in range(n):
            k, s = index[root[i]], scale[i]
            a, b = (lo[i] - t[i]) / s, (hi[i] - t[i]) / s
            if s < 0:
                a, b = b, a
            zlo[k], zhi[k] = max(zlo[k], a), min(zhi[k], b)
        # the inequalities on the representatives: one input a bound, two or more a row of G
        rows, rhs = [], []
        for coeffs, c in inequalities:
            g, h = np.zeros(K), -c
            for i, a in coeffs.items():
                g[index[root[i]]] += a * scale[i]
                h -= a * t[i]
            nonzero = np.flatnonzero(g)
            if len(nonzero) == 0:
                if h < -tol:
                    raise ValueError("joint_network: the input constraints are infeasible")
                continue
            if len(nonzero) == 1:
                k = nonzero[0]
                if g[k] > 0:
                    zhi[k] = min(zhi[k], h / g[k])
                else:
                    zlo[k] = max(zlo[k], h / g[k])
                continue
            rows.append(g)
            rhs.append(h)
        # the boxes tightened by the inequalities (bound propagation: g_k z_k <= h - the least of
        # the other terms), so that a clamp into a box cannot break an inequality
        for _ in range(PROPAGATION_ROUNDS):
            changed = False
            for g, h in zip(rows, rhs, strict=True):
                for k in np.flatnonzero(g):
                    rest = g.copy()
                    rest[k] = 0.0
                    low, _ = _interval(rest, zlo, zhi)
                    if not np.isfinite(low):
                        continue
                    bound = (h - low) / g[k]
                    if g[k] > 0 and bound < zhi[k]:
                        zhi[k], changed = bound, True
                    elif g[k] < 0 and bound > zlo[k]:
                        zlo[k], changed = bound, True
            if not changed:
                break
        if np.any(zlo > zhi):
            raise ValueError("joint_network: the range of an input is empty")
        # 3. the new inputs: independent inequality rows, then representatives with finite boxes
        M, wlo, whi, kept = [], [], [], []
        for g, h in zip(rows, rhs, strict=True):
            if np.linalg.matrix_rank(np.array(M + [g])) > len(M):
                lower, upper = _interval(g, zlo, zhi)
                M.append(g)
                wlo.append(lower)
                whi.append(min(upper, h))
                kept.append(True)
            else:
                kept.append(False)
        units = sorted(range(K), key=lambda k: not (np.isfinite(zlo[k]) and np.isfinite(zhi[k])))
        self.unit_of = {}
        for k in units:
            if len(M) == K:
                break
            e = np.eye(K)[k]
            if np.linalg.matrix_rank(np.array(M + [e])) > len(M):
                self.unit_of[k] = len(M)
                M.append(e)
                wlo.append(zlo[k])
                whi.append(zhi[k])
        self.lo, self.hi = np.array(wlo), np.array(whi)
        if not (np.isfinite(self.lo).all() and np.isfinite(self.hi).all()):
            raise ValueError("vnnlib: an input is unbounded")
        if np.any(self.lo > self.hi):
            raise ValueError("joint_network: the input constraints are infeasible")
        Minv = np.linalg.inv(np.array(M))
        # 4. the representatives that leave their boxes on the box of w: clamped
        clamps = []
        for k in range(K):
            if k in self.unit_of:
                continue
            lower, upper = _interval(Minv[k], self.lo, self.hi)
            low = zlo[k] if lower < zlo[k] else None
            high = zhi[k] if upper > zhi[k] else None
            if low is not None or high is not None:
                clamps.append((k, low, high))
        self.S, self.t, self.Minv, self.zlo, self.zhi = S, t, Minv, zlo, zhi
        self.reps = reps
        self.rows = (np.array(rows).reshape(-1, K), np.array(rhs))
        self.unenforced = int(len(kept) - sum(kept))
        self._front(clamps)

    @property
    def dim(self):
        """The number of new inputs w."""
        return self.lo.shape[0]

    def inputs(self, w):
        """The stacked inputs v (B, n) of the rows w (B, k), in float64."""
        w = np.atleast_2d(np.asarray(w, dtype=float))
        h = w if self.W0 is None else np.maximum(w @ self.W0.T + self.b0, 0.0)
        return h @ self.A.T + self.c

    def feasible(self, v, tol=0.0):
        """Whether the stacked inputs v (n,) meet the boxes, equalities and inequalities of the
        case on the representatives (up to tol)."""
        v = np.asarray(v, dtype=float)
        # the representatives: the roots' own inputs (scale 1, shift 0)
        z = v[self.reps]
        if np.any(np.abs(self.S @ z + self.t - v) > tol):
            return False
        if np.any(z < self.zlo - tol) or np.any(z > self.zhi + tol):
            return False
        G, h = self.rows
        return bool(np.all(G @ z <= h + tol))

    def _front(self, clamps):
        """The front of the joint network: h = relu(W0 w + b0) (when a representative is
        clamped, else h = w) and v = A h + c."""
        K, k = self.Minv.shape[0], self.dim
        if not clamps:
            self.W0 = self.b0 = None
            self.A, self.c = self.S @ self.Minv, self.t.copy()
            return
        # the inputs w through the ReLU unchanged (w >= lo), then one or two ReLUs per clamp
        W0, b0 = [np.eye(k)], [-self.lo]
        for row, low, high in clamps:
            for bound in (low, high):
                if bound is not None:
                    W0.append(self.Minv[row][None, :])
                    b0.append(np.array([-bound]))
        self.W0, self.b0 = np.vstack(W0), np.concatenate(b0)
        # z from h: w = h[:k] + lo; a clamped e as lo + relu(e - lo) - relu(e - hi)
        Z, z0 = np.zeros((K, self.W0.shape[0])), np.zeros(K)
        Z[:, :k] = self.Minv
        z0[:] = self.Minv @ self.lo
        at = k
        for row, low, high in clamps:
            Z[row, :] = 0.0
            if low is not None:
                # lo + relu(e - lo), and minus relu(e - hi) when bounded above too
                z0[row] = low
                Z[row, at] = 1.0
                at += 1
                if high is not None:
                    Z[row, at] = -1.0
                    at += 1
            else:
                # e - relu(e - hi): e from w
                Z[row, :k] = self.Minv[row]
                z0[row] = self.Minv[row] @ self.lo
                Z[row, at] = -1.0
                at += 1
        self.A, self.c = self.S @ Z, self.S @ z0 + self.t


def joint_layers(imap, nets, slices):
    """(layers, activations) of the joint network: the front of imap, then the networks side by
    side. nets: per network its lowered (layers [(W, b)], activations), all of the same depth and
    activations; slices: per network the slice of its inputs in the stacked inputs. The output is
    the networks' outputs stacked in their order. Weights as float64 numpy arrays."""
    acts = [list(a) for _, a in nets]
    if any(a != acts[0] for a in acts):
        raise NotImplementedError("joint_network: networks of different depths or activations")
    for layers, _ in nets:
        if any(isinstance(layer, dict) for layer in layers):
            raise NotImplementedError("joint_network: networks of dense layers are supported")
    weights = [[(_array(W), _array(b)) for W, b in layers] for layers, _ in nets]
    out = []
    if imap.W0 is not None:
        out.append((imap.W0, imap.b0))
    # the first layer of each network reads its slice of v = A h + c
    first_W, first_b = [], []
    for (W, b), sl in zip((w[0] for w in weights), slices, strict=True):
        A, c = imap.A[sl], imap.c[sl]
        if W.shape[1] != A.shape[0]:
            raise ValueError(f"joint_network: a network of {W.shape[1]} inputs, {A.shape[0]} given")
        first_W.append(W @ A)
        first_b.append(b + W @ c)
    out.append((np.vstack(first_W), np.concatenate(first_b)))
    # the later layers block-diagonal
    for k in range(1, len(weights[0])):
        out.append((_block_diag([w[k][0] for w in weights]),
                    np.concatenate([w[k][1] for w in weights])))
    activations = (["relu"] if imap.W0 is not None else []) + acts[0]
    return out, activations


# -----------------------------------------  AUXILIARY  ------------------------------------------ #

PROPAGATION_ROUNDS = 8  # rounds of bound propagation through the inequalities, at most


def _array(t):
    """A torch tensor or an array as a float64 numpy array."""
    return np.asarray(t.detach().cpu().numpy() if hasattr(t, "detach") else t, dtype=float)


def _block_diag(blocks):
    """The block-diagonal matrix of the blocks."""
    rows, cols = sum(b.shape[0] for b in blocks), sum(b.shape[1] for b in blocks)
    out, r, c = np.zeros((rows, cols)), 0, 0
    for b in blocks:
        out[r : r + b.shape[0], c : c + b.shape[1]] = b
        r, c = r + b.shape[0], c + b.shape[1]
    return out


def _interval(g, lo, hi):
    """[min, max] of g . z over the box [lo, hi] (a zero coefficient of an infinite bound: 0)."""
    lower = sum(a * (lo[i] if a > 0 else hi[i]) for i, a in enumerate(g) if a != 0)
    upper = sum(a * (hi[i] if a > 0 else lo[i]) for i, a in enumerate(g) if a != 0)
    return float(lower), float(upper)


def _split_relations(relations):
    """(equalities, inequalities) of relations [({i: a}, c)]: a relation and its opposite (both
    halves of an ==) are one equality; zero coefficients dropped."""
    def key(coeffs, c):
        return tuple(sorted((i, a + 0.0) for i, a in coeffs.items() if a != 0)), c + 0.0

    rels = [({i: a for i, a in r.items() if a != 0}, float(c)) for r, c in relations]
    keys = [key(r, c) for r, c in rels]
    present = set(keys)
    used, equalities, inequalities = set(), [], []
    for (coeffs, c), k in zip(rels, keys, strict=True):
        if k in used:
            continue
        opposite = key({i: -a for i, a in coeffs.items()}, -c)
        if opposite in present and opposite != k and opposite not in used:
            used.update((k, opposite))
            equalities.append((coeffs, c))
        else:
            inequalities.append((coeffs, c))
    return equalities, inequalities


def _both(equality):
    """The two inequalities of an equality sum a x + c == 0."""
    coeffs, c = equality
    return [(coeffs, c), ({i: -a for i, a in coeffs.items()}, -c)]


def _union(n, equalities, tol):
    """(root, scale, shift) per input: x_i = scale_i x_root_i + shift_i from the equalities
    a_i x_i + a_j x_j + c == 0 (a union-find with affine maps; the lower index the root)."""
    parent, scale, shift = list(range(n)), [1.0] * n, [0.0] * n

    def find(i):
        # the root of i and x_i = s x_root + t (the path not compressed: few equalities)
        s, t = 1.0, 0.0
        while parent[i] != i:
            s, t = s * scale[i], s * shift[i] + t
            i = parent[i]
        return i, s, t

    for coeffs, c in equalities:
        (i, ai), (j, aj) = sorted(coeffs.items())
        # x_j = s x_i + t
        s, t = -ai / aj, -c / aj
        ri, si, ti = find(i)
        rj, sj, tj = find(j)
        if ri == rj:
            # x_j = sj x_r + tj must be s (si x_r + ti) + t
            if abs(sj - s * si) > tol or abs(tj - (s * ti + t)) > tol:
                raise NotImplementedError("joint_network: equalities that fix an input")
            continue
        # the higher root under the lower: x_rj = (x_j - tj) / sj with x_j = s (si x_ri + ti) + t
        if rj > ri:
            parent[rj], scale[rj], shift[rj] = ri, s * si / sj, (s * ti + t - tj) / sj
        else:
            # x_ri = (x_i - ti) / si with x_i = (x_j - t) / s = (sj x_rj + tj - t) / s
            parent[ri], scale[ri], shift[ri] = rj, sj / (s * si), ((tj - t) / s - ti) / si
    out = [find(i) for i in range(n)]
    return [r for r, _, _ in out], [s for _, s, _ in out], [t for _, _, t in out]


# ----------------------------------------  END OF CODE  ----------------------------------------- #
