"""vnnlib - a VNN-LIB 1.0 or 2.0 property as input boxes and disjunctions of output polytopes

A property is a list of cases; a case is an input box [lo, hi] and the unsafe output set, a union
of polytopes {y : A y <= b}. The property is violated (sat) if some input of a box maps into the
union of its case. A 2.0 property may declare several networks (VNN-COMP 2026's equal-to and
isomorphic-to pairs): their inputs and their outputs are stacked in the order of declaration
(Property.networks gives the offsets), and the input constraints that relate inputs (X_f[i] ==
X_g[i], X_g[0] <= X_f[0]) are the case's relations, sum_i a_i x_i + c <= 0 (cora.joint_network
turns them into a box of new inputs). VNN-LIB 1.0 is read with the semantics of the VNN-COMP
scoring parser (vnncomp2026_results/SCORING/vnnlib_v1.py: box asserts, one top-level disjunction,
cases of the same box merged); VNN-LIB 2.0 by the reference parser (the vnnlib package), its
assertions multiplied out into the same cases. Inputs and outputs are indexed flat, in C order of
their ONNX shapes.

Syntax:   prop = read_vnnlib("prop.vnnlib", num_inputs=5, num_outputs=5)
          prop.cases[0].lo, .hi, .polytopes          float64 arrays; [(A (k, m), b (k,))]
          prop.specifications(0)                     cora.Specification.unsafe_set per polytope
          prop.cases[0].relations                    [({input: a_i}, c)] of sum a_i x_i + c <= 0
          prop.networks                              [NetworkInfo] of a 2.0 query (None for 1.0)
          text = to_vnnlib2(v1_text, (1, 5), (1, 5)) the 2.0 form of a 1.0 file (as VNN-COMP's)
See also: cora.onnx_network, cora.counterexample, docs/nn_verification.md

Authors:       Claude Opus 5.5
Written:       21-November-2026
Last update:   25-November-2026 (several networks, relations of inputs)
Reviewed:      ---
"""

import gzip
import math
import re
from dataclasses import dataclass, field
from decimal import Decimal

import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

@dataclass
class Case:
    """An input box [lo, hi] and the unsafe outputs: the union of the polytopes A y <= b."""

    lo: np.ndarray
    hi: np.ndarray
    polytopes: list = field(default_factory=list)
    relations: list = field(default_factory=list)  # [({input: a}, c)]: sum a x + c <= 0


@dataclass
class NetworkInfo:
    """A network of a 2.0 query: its name, its inputs and outputs as (name, shape, offset) in the
    stacked inputs and outputs, and the network it is equal-to or isomorphic-to (or "")."""

    name: str
    inputs: list
    outputs: list
    equal_to: str = ""
    isomorphic_to: str = ""

    @property
    def num_inputs(self):
        return sum(math.prod(shape) for _, shape, _ in self.inputs)

    @property
    def num_outputs(self):
        return sum(math.prod(shape) for _, shape, _ in self.outputs)


@dataclass
class Property:
    """A VNN-LIB property: its version (1 or 2), its cases, and the 2.0 query (None for 1.0)."""

    version: int
    num_inputs: int
    num_outputs: int
    cases: list
    query: object = None
    networks: list = None  # [NetworkInfo] of a 2.0 query

    def specifications(self, case=0):
        """The unsafe sets of a case as cora.Specification objects (one per polytope)."""
        import cora

        return [
            cora.Specification.unsafe_set(cora.Polytope(A, b))
            for A, b in self.cases[case].polytopes
        ]

    def stacked(self):
        """All polytopes of all cases as arrays padded to one row count: case index (P,), A
        (P, K, m), b (P, K); a padding row is 0 <= +inf, which never decides."""
        rows = max(A.shape[0] for c in self.cases for A, _ in c.polytopes)
        index, As, bs = [], [], []
        for k, case in enumerate(self.cases):
            for A, b in case.polytopes:
                index.append(k)
                As.append(np.vstack([A, np.zeros((rows - A.shape[0], self.num_outputs))]))
                bs.append(np.concatenate([b, np.full(rows - b.shape[0], np.inf)]))
        return np.array(index), np.array(As), np.array(bs)


def read_vnnlib(path, num_inputs, num_outputs):
    """The property of a VNN-LIB file (.vnnlib or .vnnlib.gz); the version from its header."""
    return parse_vnnlib(read_text(path), num_inputs, num_outputs)


def parse_vnnlib(text, num_inputs, num_outputs):
    """The property of a VNN-LIB text; for 2.0, num_inputs and num_outputs None take the stacked
    inputs and outputs of the declared networks."""
    if VERSION_2.search(text):
        cases, query, networks = _read_v2(text, num_inputs, num_outputs)
        num_inputs = sum(net.num_inputs for net in networks) if num_inputs is None else num_inputs
        num_outputs = (sum(net.num_outputs for net in networks) if num_outputs is None
                       else num_outputs)
        prop = Property(2, num_inputs, num_outputs, cases, query, networks)
    else:
        prop = Property(1, num_inputs, num_outputs, _read_v1(text, num_inputs, num_outputs))
    for case in prop.cases:
        if case.lo.shape != (num_inputs,):
            raise ValueError(f"vnnlib: {case.lo.shape[0]} inputs, the network has {num_inputs}")
    return prop


def to_vnnlib2(text, input_shape, output_shape, input_type="float32", output_type="float32"):
    """The VNN-LIB 2.0 form of a 1.0 text, as VNN-COMP's benchmark repository writes it: a
    declare-network of the ONNX shapes, X_i and Y_j as indices into them, the rest kept."""

    def index(match):
        name, flat = match.group(1), int(match.group(2))
        shape = input_shape if name == "X" else output_shape
        return f"{name}[{','.join(str(int(i)) for i in np.unravel_index(flat, shape))}]"

    dims = lambda s: "[" + ", ".join(str(d) for d in s) + "]"  # noqa: E731
    head = [
        "(vnnlib-version <2.0>)",
        "",
        "(declare-network N",
        f"    (declare-input  X {input_type} {dims(input_shape)})",
        f"    (declare-output Y {output_type} {dims(output_shape)})",
        ")",
    ]
    # the declarations of 1.0 go; every X_i and Y_j becomes an index
    body = [line for line in text.splitlines() if not DECLARE.match(line.strip())]
    body = [re.sub(r"\b([XY])_(\d+)\b", index, line) for line in body]
    # an integer constant is a real one in 2.0 (its type checker refuses 1 for a float32)
    body = [
        line
        if line.lstrip().startswith(";")
        else re.sub(r"(?<=[\s(])(-?\d+)(?=[\s)])", r"\1.0", line)
        for line in body
    ]
    # an exponent is no 2.0 number: the same digits positionally (1e-06: 0.000001)
    return positional("\n".join(head + body) + "\n")


def positional(text):
    """The text with every number in exponent form (2.1e-05) written positionally (0.000021), the
    same digits (Decimal, no rounding), as VNN-COMP's 2.0 files write it; comment lines kept."""

    def plain(match):
        out = format(Decimal(match.group(0)), "f")
        return out if "." in out else out + ".0"

    lines = text.split("\n")
    return "\n".join(
        line if line.lstrip().startswith(";") else EXPONENT.sub(plain, line) for line in lines
    )


# -----------------------------------------  AUXILIARY  ------------------------------------------ #


COMPARISON = r"\((<=|>=) (\S+) (\S+)\)"

DECLARE = re.compile(r"^\(declare-const (X|Y)_(\S+) Real\)$")

SIMPLE_ASSERT = re.compile(r"^\(assert " + COMPARISON + r"\)$")

DNF_ASSERT = re.compile(r"^\(assert \(or (\(and ?(" + COMPARISON + r")+\))+\)\)$")

EXPONENT = re.compile(r"(?<![\w.\[,])-?\d+(\.\d*)?[eE][-+]?\d+(?![\w.])")

VERSION_2 = re.compile(r"\(\s*vnnlib-version\s*<\s*2\.0\s*>\s*\)")


def read_text(path):
    """The text of a file, gzipped or not."""
    if str(path).endswith(".gz"):
        with gzip.open(path, "rt", encoding="utf-8") as stream:
            return stream.read()
    with open(path, encoding="utf-8") as stream:
        return stream.read()


def statements(text):
    """The statements of a VNN-LIB 1.0 text: comments dropped, one statement per balanced
    parenthesis, whitespace normalized (as the scoring's read_statements)."""
    out, current, depth = [], "", 0
    for line in text.splitlines():
        line = line.split(";", 1)[0].strip()
        if not line:
            continue
        depth += line.count("(") - line.count(")")
        if depth < 0:
            raise ValueError("vnnlib: mismatched parentheses")
        current += (" " if current else "") + line
        if depth == 0:
            out.append(current)
            current = ""
    if current:
        out.append(current)
    return [" ".join(s.split()).replace("( ", "(").replace(") ", ")") for s in out]


class _Case1:
    """A case while a 1.0 file is read: the box as [lo, hi] lists and the rows of A y <= b."""

    def __init__(self, n):
        self.lo, self.hi, self.rows, self.rhs = [-np.inf] * n, [np.inf] * n, [], []

    def copy(self):
        other = _Case1(0)
        other.lo, other.hi, other.rows, other.rhs = (
            list(self.lo),
            list(self.hi),
            list(self.rows),
            list(self.rhs),
        )
        return other

    def add(self, op, first, second, m):
        """The constraint (op first second): a bound of an input or a row on the outputs."""
        if first.startswith("X_"):
            i = int(first[2:])
            if second.startswith(("X", "Y")):
                raise ValueError(f"vnnlib: input constraints must be box ({op} {first} {second})")
            if op == "<=":
                self.hi[i] = min(float(second), self.hi[i])
            else:
                self.lo[i] = max(float(second), self.lo[i])
            if self.lo[i] > self.hi[i]:
                raise ValueError(f"vnnlib: {first} range is empty")
            return
        if op == ">=":
            first, second = second, first
        row, rhs = [0.0] * m, 0.0
        if first.startswith("Y_") and second.startswith("Y_"):
            row[int(first[2:])], row[int(second[2:])] = 1.0, -1.0
        elif first.startswith("Y_"):
            row[int(first[2:])], rhs = 1.0, float(second)
        else:
            row[int(second[2:])], rhs = -1.0, -float(first)
        self.rows.append(row)
        self.rhs.append(rhs)


def _read_v1(text, n, m):
    """The cases of a VNN-LIB 1.0 text (the scoring's read_vnnlib_simple)."""
    cases = [_Case1(n)]
    for s in statements(text):
        if DECLARE.match(s):
            continue
        simple = SIMPLE_ASSERT.match(s)
        if simple:
            for case in cases:
                case.add(*simple.groups(), m)
            continue
        if not DNF_ASSERT.match(s):
            raise ValueError(f"vnnlib: cannot parse the statement {s[:80]}")
        # each conjunct of the disjunction copies every case so far
        tokens = s.replace("(", " ").replace(")", " ").split()[2:]
        conjuncts = " ".join(tokens).split("and")[1:]
        old, cases = cases, []
        for case in old:
            for conjunct in conjuncts:
                copy = case.copy()
                words = conjunct.split()
                for k in range(len(words) // 3):
                    copy.add(*words[3 * k : 3 * k + 3], m)
                cases.append(copy)
    # cases of the same box are one case with several polytopes
    merged = {}
    for case in cases:
        key = (tuple(case.lo), tuple(case.hi))
        merged.setdefault(key, []).append(
            (np.array(case.rows, dtype=float).reshape(-1, m), np.array(case.rhs, dtype=float))
        )
    out = []
    for (lo, hi), polytopes in merged.items():
        lo, hi = np.array(lo), np.array(hi)
        if not (np.isfinite(lo).all() and np.isfinite(hi).all()):
            raise ValueError("vnnlib: an input is unbounded")
        out.append(Case(lo, hi, polytopes))
    return out


def _read_v2(text, n, m):
    """The cases, the query and the networks of a VNN-LIB 2.0 text: parsed and type-checked by the
    reference parser, each assertion in disjunctive normal form, the conjunction of the assertions
    multiplied out, and every conjunct split into its input box, its relations of inputs and its
    output polytope. n, m: the stacked inputs and outputs (None: those of the query)."""
    import vnnlib

    # the reference parser refuses an exponent (1e-05): positional, as VNN-COMP's own 2.0 files
    query = vnnlib.parse_query_string(positional(text))
    if any(net.hidden for net in query.networks):
        raise NotImplementedError("vnnlib: networks without hidden variables are supported")
    networks, offsets = _networks(query)
    n = sum(net.num_inputs for net in networks) if n is None else n
    m = sum(net.num_outputs for net in networks) if m is None else m
    conjuncts = [[]]
    for assertion in query.assertions:
        dnf = assertion.expr.to_dnf()
        if len(dnf) == 1:
            # one conjunct (a box's bound): appended in place, not every list copied (vggnet16's
            # 301057 assertions made that quadratic)
            for c in conjuncts:
                c.extend(dnf[0])
            continue
        conjuncts = [c + list(d) for c in conjuncts for d in dnf]
    # each atom's linear form once (a box atom is shared by every conjunct of the disjunction)
    forms = {}
    merged = {}
    for atoms in conjuncts:
        case = _Case1(n)
        case.relations = []
        for atom in atoms:
            if id(atom) not in forms:
                forms[id(atom)] = [_atom_form(*a) for a in _linear_atoms(atom, n, m, offsets)]
            for form in forms[id(atom)]:
                _add_atom(case, form)
        key = (tuple(case.lo), tuple(case.hi),
               tuple((tuple(sorted(r.items())), c) for r, c in case.relations))
        merged.setdefault(key, (case.relations, []))[1].append(
            (np.array(case.rows, dtype=float).reshape(-1, m), np.array(case.rhs, dtype=float))
        )
    cases = [Case(np.array(lo), np.array(hi), p, rel) for (lo, hi, _), (rel, p) in merged.items()]
    # an input bounded only through a relation (X_g[0] <= X_f[0]) is checked by cora.joint_network
    bounded = (np.isfinite(c.lo).all() and np.isfinite(c.hi).all()
               for c in cases if not c.relations)
    if not all(bounded):
        raise ValueError("vnnlib: an input is unbounded")
    return cases, query, networks


def _networks(query):
    """The NetworkInfo of each network of the query, and the offset of each variable by name in
    the stacked inputs or outputs."""
    networks, offsets, at = [], {}, {"inputs": 0, "outputs": 0}
    for net in query.networks:
        lists = {}
        for kind in ("inputs", "outputs"):
            lists[kind] = []
            for d in getattr(net, kind):
                shape = tuple(int(s) for s in d.shape)
                lists[kind].append((d.name, shape, at[kind]))
                offsets[d.name] = at[kind]
                at[kind] += math.prod(shape)
        networks.append(NetworkInfo(net.name, lists["inputs"], lists["outputs"],
                                    net.equal_to or "", net.isometric_to or ""))
    return networks, offsets


def _linear_atoms(atom, n, m, offsets=None):
    """A comparison as (op, coefficients, constant, 'input' or 'output'): row . v + constant op 0
    with op '<=' (or two of them for '=='); an input's row as {index: coefficient}. offsets: the
    offset of each variable in the stacked inputs or outputs (of several networks)."""
    name = type(atom).__name__
    if name == "NotEqual":
        raise NotImplementedError("vnnlib: != is not supported")
    import vnnlib

    try:
        lhs, rhs = atom.lhs.to_linear_expr(), atom.rhs.to_linear_expr()
    except vnnlib.VNNLibException as error:  # a product of variables
        raise NotImplementedError(f"vnnlib: a nonlinear constraint ({error})") from None
    coeffs, kinds = {}, set()
    for sign, expr in ((1.0, lhs), (-1.0, rhs)):
        for term in expr.terms:
            var = term.var
            kind = str(var.kind).split(".")[-1]
            flat = int(np.ravel_multi_index(tuple(var.indices), tuple(var.shape)))
            flat += offsets.get(var.name, 0) if offsets else 0
            coeffs[(kind, flat)] = coeffs.get((kind, flat), 0.0) + sign * term.coeff
            kinds.add(kind)
    const = lhs.constant - rhs.constant
    if len(kinds) != 1 or kinds - {"Input", "Output"}:
        raise NotImplementedError("vnnlib: a constraint that mixes inputs and outputs")
    kind = kinds.pop()
    # an input's row as its nonzero coefficients {index: c} (a dense row per atom of a box of
    # vggnet16's 150528 inputs costs n^2: 734 s), an output's row dense
    if kind == "Input":
        row = {flat: c for (_, flat), c in coeffs.items() if c != 0}
        neg = {i: -c for i, c in row.items()}
    else:
        row = np.zeros(m)
        for (_, flat), c in coeffs.items():
            row[flat] = c
        neg = -row
    if name in ("LessEqual", "LessThan"):
        return [("<=", row, const, kind)]
    if name in ("GreaterEqual", "GreaterThan"):
        return [("<=", neg, -const, kind)]
    return [("<=", row, const, kind), ("<=", neg, -const, kind)]


def _atom_form(op, row, const, kind):
    """row . v + const <= 0 as ("Output", row, rhs) of A y <= b, ("Input", i, bound, upper), or
    ("Relation", row, const) of an input constraint on several inputs."""
    del op  # always "<=" (_linear_atoms)
    if kind == "Output":
        return ("Output", list(row), -const)
    if len(row) != 1:
        return ("Relation", dict(row), float(const))
    ((i, a),) = row.items()
    return ("Input", int(i), -const / float(a), a > 0)


def _add_atom(case, form):
    """An atom's form into the case: a row of A y <= b, or a bound of an input."""
    if form[0] == "Output":
        case.rows.append(form[1])
        case.rhs.append(form[2])
        return
    if form[0] == "Relation":
        case.relations.append((form[1], form[2]))
        return
    _, i, bound, upper = form
    if upper:
        case.hi[i] = min(case.hi[i], bound)
    else:
        case.lo[i] = max(case.lo[i], bound)
    if case.lo[i] > case.hi[i]:
        raise ValueError(f"vnnlib: the range of input {i} is empty")

# ----------------------------------------  END OF CODE  ----------------------------------------- #
