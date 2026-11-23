"""vnnlib - a VNN-LIB 1.0 or 2.0 property as input boxes and disjunctions of output polytopes

A property is a list of cases; a case is an input box [lo, hi] and the unsafe output set, a union
of polytopes {y : A y <= b}. The property is violated (sat) if some input of a box maps into the
union of its case. VNN-LIB 1.0 is read with the semantics of the VNN-COMP scoring parser
(vnncomp2026_results/SCORING/vnnlib_v1.py: box asserts, one top-level disjunction, cases of the
same box merged); VNN-LIB 2.0 by the reference parser (the vnnlib package), its assertions
multiplied out into the same cases. Inputs and outputs are indexed flat, in C order of their ONNX
shapes.

Syntax:   prop = read_vnnlib("prop.vnnlib", num_inputs=5, num_outputs=5)
          prop.cases[0].lo, .hi, .polytopes          float64 arrays; [(A (k, m), b (k,))]
          prop.specifications(0)                     cora.Specification.unsafe_set per polytope
          text = to_vnnlib2(v1_text, (1, 5), (1, 5)) the 2.0 form of a 1.0 file (as VNN-COMP's)
See also: cora.onnx_network, cora.counterexample, docs/nn_verification.md

Authors:       Claude Opus 5.5
Written:       21-November-2026
Last update:   ---
Reviewed:      ---
"""

import gzip
import re
from dataclasses import dataclass, field

import numpy as np


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

@dataclass
class Case:
    """An input box [lo, hi] and the unsafe outputs: the union of the polytopes A y <= b."""

    lo: np.ndarray
    hi: np.ndarray
    polytopes: list = field(default_factory=list)


@dataclass
class Property:
    """A VNN-LIB property: its version (1 or 2), its cases, and the 2.0 query (None for 1.0)."""

    version: int
    num_inputs: int
    num_outputs: int
    cases: list
    query: object = None

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
    """The property of a VNN-LIB text."""
    if VERSION_2.search(text):
        cases, query = _read_v2(text, num_inputs, num_outputs)
        prop = Property(2, num_inputs, num_outputs, cases, query)
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
    # an exponent is no 2.0 number: the shortest decimal of the same double (1e-06: 0.000001)
    body = [
        line
        if line.lstrip().startswith(";")
        else re.sub(r"(?<=[\s(])(-?\d+(?:\.\d*)?[eE][-+]?\d+)(?=[\s)])", _decimal, line)
        for line in body
    ]
    # an integer constant is a real one in 2.0 (its type checker refuses 1 for a float32)
    body = [
        line
        if line.lstrip().startswith(";")
        else re.sub(r"(?<=[\s(])(-?\d+)(?=[\s)])", r"\1.0", line)
        for line in body
    ]
    return "\n".join(head + body) + "\n"


# -----------------------------------------  AUXILIARY  ------------------------------------------ #


def _decimal(match):
    """The number of a match in positional notation, the shortest that parses to the same double."""
    return np.format_float_positional(float(match.group(1)), unique=True, trim="0")

COMPARISON = r"\((<=|>=) (\S+) (\S+)\)"

DECLARE = re.compile(r"^\(declare-const (X|Y)_(\S+) Real\)$")

SIMPLE_ASSERT = re.compile(r"^\(assert " + COMPARISON + r"\)$")

DNF_ASSERT = re.compile(r"^\(assert \(or (\(and ?(" + COMPARISON + r")+\))+\)\)$")

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
    """The cases and the query of a VNN-LIB 2.0 text: parsed and type-checked by the reference
    parser, each assertion in disjunctive normal form, the conjunction of the assertions
    multiplied out, and every conjunct split into its input box and its output polytope."""
    import vnnlib

    query = vnnlib.parse_query_string(text)
    if len(query.networks) != 1 or query.networks[0].hidden:
        raise NotImplementedError("vnnlib: one network without hidden variables is supported")
    conjuncts = [[]]
    for assertion in query.assertions:
        dnf = assertion.expr.to_dnf()
        conjuncts = [c + list(d) for c in conjuncts for d in dnf]
    # each atom's linear form once (a box atom is shared by every conjunct of the disjunction)
    forms = {}
    merged = {}
    for atoms in conjuncts:
        case = _Case1(n)
        for atom in atoms:
            if id(atom) not in forms:
                forms[id(atom)] = [_atom_form(*a) for a in _linear_atoms(atom, n, m)]
            for form in forms[id(atom)]:
                _add_atom(case, form)
        key = (tuple(case.lo), tuple(case.hi))
        merged.setdefault(key, []).append(
            (np.array(case.rows, dtype=float).reshape(-1, m), np.array(case.rhs, dtype=float))
        )
    cases = [Case(np.array(lo), np.array(hi), p) for (lo, hi), p in merged.items()]
    if not all(np.isfinite(c.lo).all() and np.isfinite(c.hi).all() for c in cases):
        raise ValueError("vnnlib: an input is unbounded")
    return cases, query


def _linear_atoms(atom, n, m):
    """A comparison as (op, coefficients, constant, 'input' or 'output'): row . v + constant op 0
    with op '<=' (or two of them for '==')."""
    name = type(atom).__name__
    if name == "NotEqual":
        raise NotImplementedError("vnnlib: != is not supported")
    lhs, rhs = atom.lhs.to_linear_expr(), atom.rhs.to_linear_expr()
    coeffs, kinds = {}, set()
    for sign, expr in ((1.0, lhs), (-1.0, rhs)):
        for term in expr.terms:
            var = term.var
            kind = str(var.kind).split(".")[-1]
            flat = int(np.ravel_multi_index(tuple(var.indices), tuple(var.shape)))
            coeffs[(kind, flat)] = coeffs.get((kind, flat), 0.0) + sign * term.coeff
            kinds.add(kind)
    const = lhs.constant - rhs.constant
    if len(kinds) != 1 or kinds - {"Input", "Output"}:
        raise NotImplementedError("vnnlib: a constraint that mixes inputs and outputs")
    kind = kinds.pop()
    row = np.zeros(n if kind == "Input" else m)
    for (_, flat), c in coeffs.items():
        row[flat] = c
    if name in ("LessEqual", "LessThan"):
        return [("<=", row, const, kind)]
    if name in ("GreaterEqual", "GreaterThan"):
        return [("<=", -row, -const, kind)]
    return [("<=", row, const, kind), ("<=", -row, -const, kind)]


def _atom_form(op, row, const, kind):
    """row . v + const <= 0 as ("Output", row, rhs) of A y <= b, or ("Input", i, bound, upper)."""
    del op  # always "<=" (_linear_atoms)
    if kind == "Output":
        return ("Output", list(row), -const)
    nonzero = np.flatnonzero(row)
    if len(nonzero) != 1:
        raise NotImplementedError("vnnlib: input constraints must be box constraints")
    i, a = int(nonzero[0]), float(row[nonzero[0]])
    return ("Input", i, -const / a, a > 0)


def _add_atom(case, form):
    """An atom's form into the case: a row of A y <= b, or a bound of an input."""
    if form[0] == "Output":
        case.rows.append(form[1])
        case.rhs.append(form[2])
        return
    _, i, bound, upper = form
    if upper:
        case.hi[i] = min(case.hi[i], bound)
    else:
        case.lo[i] = max(case.lo[i], bound)
    if case.lo[i] > case.hi[i]:
        raise ValueError(f"vnnlib: the range of input {i} is empty")

# ----------------------------------------  END OF CODE  ----------------------------------------- #
