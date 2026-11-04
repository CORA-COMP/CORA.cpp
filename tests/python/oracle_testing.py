"""oracle_testing - what a Python test against the CORALean oracle shares (tests/global/oracle/oracle_testing.h)

Syntax:   @oracle_testing.needs_oracle
          class SupportFunction(unittest.TestCase):
              def test_zonotope(self):
                  def body(rnd):
                      Z, d = rnd.zonotope(3, 5), rnd.direction(3)
                      oracle_testing.check(Z.support_function(d), cora.lean.Zonotope(Z).support_function(d))
                  oracle_testing.run("zonotope.supportFunc", body)
Modes:    run calls the body twice with the same random operands: "exact" (CORA.cpp on the lean
          backend against the oracle at dtype nearest: bit for bit) and "sound" (CORA.cpp on its
          default backend against the sound oracle at binary64: inside its enclosure); check does
          the check of the mode, check_exact and check_enclosed one of them.
Results:  CORA.cpp's result and the oracle's, as float, array, cora.Interval / lean.Interval,
          cora.Zonotope / lean.Zonotope, bool and lean.Certified (a proof only when `proved`), a
          list of indices, a pair of halves or None.
Skip:     needs_oracle skips a test unless CORACPP_ORACLE and CORACPP_ORACLE_UNSOUND are set.
"""

import contextlib
import os
import unittest
import warnings

import cora
import numpy as np

AVAILABLE = bool(os.environ.get("CORACPP_ORACLE") and os.environ.get("CORACPP_ORACLE_UNSOUND"))
needs_oracle = unittest.skipUnless(AVAILABLE, "needs CORACPP_ORACLE and CORACPP_ORACLE_UNSOUND")

_state = {"mode": "exact", "op": ""}


class Random:
    """Operands from a fixed seed: normal numbers, which are exact doubles."""

    def __init__(self, seed):
        self.gen = np.random.default_rng(seed)

    def scalar(self):
        return float(self.gen.standard_normal())

    def vector(self, n):
        return self.gen.standard_normal(n)

    def matrix(self, rows, cols):
        return self.gen.standard_normal((rows, cols))

    def direction(self, n):
        return self.vector(n)

    def zonotope(self, n, m):
        return cora.Zonotope(self.vector(n), self.matrix(n, m))

    def box(self, n):
        c, r = self.vector(n), self.gen.uniform(0, 1, n)
        return cora.Interval(c - r, c + r)


# Reading results -------------------------------------------------------------------------------

def _array(x):
    """A result as a float64 array, a column as a vector."""
    a = np.asarray(x.gather() if isinstance(x, cora.lean.Tensor) else x, dtype=np.float64)
    return a.reshape(a.shape[:-1]) if a.ndim == 2 and a.shape[1] == 1 else a


def _bits(a):
    return np.ascontiguousarray(a).view(np.uint64)


def _same_bits(what, a, b):
    """Raises with the first difference unless a and b have one shape and the same bits."""
    a, b = _array(a), _array(b)
    if a.shape != b.shape:
        raise AssertionError(f"{what}: shape {a.shape} in CORA.cpp, {b.shape} in the oracle")
    differ = np.flatnonzero(_bits(a).reshape(-1) != _bits(b).reshape(-1))
    if differ.size:
        i = differ[0]
        raise AssertionError(
            f"{what} [{i}]: {a.reshape(-1)[i]!r} (0x{_bits(a).reshape(-1)[i]:x}) in CORA.cpp, "
            f"{b.reshape(-1)[i]!r} (0x{_bits(b).reshape(-1)[i]:x}) in the oracle")


def _parts(x):
    """The arrays that make a set (and the error box of an oracle's zonotope is left out)."""
    if isinstance(x, cora.lean.Zonotope):
        x = x.gather()[0]
    if isinstance(x, cora.lean.Interval):
        x = x.gather()
    if isinstance(x, (cora.Fullspace, cora.lean.Fullspace)):
        return {"dim": np.array([float(x.dim)])}
    if isinstance(x, cora.Zonotope):
        return {"c": x.center, "G": x.generators}
    if isinstance(x, cora.Interval):
        return {"inf": x.inf, "sup": x.sup}
    return None


def same_bits(what, a, b):
    """Raises unless a (CORA.cpp's result) and b (the oracle's) are the same, bit for bit."""
    if isinstance(b, cora.lean.Certified):
        b = b.value
    if isinstance(b, (tuple, list)):
        if len(a) != len(b):
            raise AssertionError(f"{what}: {len(a)} results in CORA.cpp, {len(b)} in the oracle")
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            same_bits(f"{what}.{i + 1}", x, y)
    elif a is None or b is None:
        if (a is None) != (b is None):
            raise AssertionError(f"{what}: {'none' if a is None else 'a result'} in CORA.cpp, "
                                 f"{'none' if b is None else 'a result'} in the oracle")
    elif isinstance(b, (bool, np.bool_)):
        if bool(a) != bool(b):
            raise AssertionError(f"{what}: {a} in CORA.cpp, {b} in the oracle")
    elif _parts(b) is not None:
        pa, pb = _parts(a), _parts(b)
        for key in pb:
            _same_bits(f"{what}.{key}", pa[key], pb[key])
    else:
        _same_bits(what, a, b)


# Enclosure checks ------------------------------------------------------------------------------

def enclosed(what, a, b):
    """Raises unless a (CORA.cpp's result) lies in b (the sound oracle's)."""
    if isinstance(b, cora.lean.Certified):
        if b.proved and bool(a) != b.value:
            raise AssertionError(f"{what}: CORA.cpp says {a}, the oracle proves {b.value}")
    elif isinstance(b, (bool, np.bool_)):
        if b and not a:
            raise AssertionError(f"{what}: CORA.cpp says {a}, the oracle says {b}")
    elif isinstance(b, tuple):
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            enclosed(f"{what}.{i + 1}", x, y)
    elif isinstance(b, cora.lean.Fullspace):
        same_bits(what, a, b)  # the whole space has no enclosure but itself
    elif isinstance(b, cora.lean.Interval):
        lo, hi = _array(b.inf), _array(b.sup)
        inf, sup = _array(a.inf), _array(a.sup)
        if not (np.all(lo <= inf) and np.all(sup <= hi)):
            raise AssertionError(f"{what}: [{inf}, {sup}] is not inside [{lo}, {hi}]")
    elif isinstance(b, cora.lean.Zonotope):
        hull = b.enclosure()
        if not hull.contains(cora.lean.Zonotope(a)).proved:
            enclosed(f"{what} (hull)", a.interval(), hull.interval())
    elif isinstance(b, (float, np.floating)):
        if float(np.asarray(a).reshape(-1)[0]) > b:
            raise AssertionError(f"{what}: {float(a)!r} exceeds the bound {b!r}")
    # a point-valued result (a tensor, indices) has no enclosure


def mode():
    """The mode of the running `run`: exact or sound."""
    return _state["mode"]


def check_exact(op, cora_result, lean_result):
    """CORA.cpp's result equals the oracle's bit for bit (run it on the lean backend at nearest)."""
    same_bits(op, cora_result, lean_result)


def check_enclosed(op, cora_result, sound_result):
    """CORA.cpp's result lies in the sound oracle's."""
    enclosed(op, cora_result, sound_result)


def check(cora_result, lean_result):
    """The check of the running mode, for the op of `run`."""
    (check_exact if _state["mode"] == "exact" else check_enclosed)(_state["op"], cora_result, lean_result)


def run(op, body, seed=7):
    """Calls body(Random) in both modes with the same seed: exact (lean backend, dtype nearest)
    and sound (default backend, dtype binary64); what the body builds follows the mode."""
    _state["op"] = op
    for mode, backend, dtype in (("exact", "lean", "nearest"),
                                 ("sound", cora.current_backend(), "binary64")):
        _state["mode"] = mode
        with contextlib.ExitStack() as scopes:
            scopes.enter_context(warnings.catch_warnings())
            warnings.simplefilter("ignore", cora.CORAWarning)  # nearest has no guarantee, by design
            scopes.enter_context(cora.backend(backend))
            scopes.enter_context(cora.lean.dtype(dtype))
            body(Random(seed))
