"""matlab - the bridge for MATLAB CORA (toPy, fromPy), which cannot call pybind11 classes

MATLAB cannot take a pybind11 class itself (cora.Zonotope) but works with its instances, so these
functions build the objects. Data comes as python lists or numpy arrays. The names and the
positional calls are the ones MATLAB CORA's toPy/fromPy and examples/matlab use (CORA's names:
intervalMatrix, linearSys, reach(X0, timeStep, tFinal, ...), outputSet, timeInt, timePoint); they
map onto the keyword API of the package and compute nothing themselves.

    m = py.importlib.import_module('cora.matlab')
    Zpy = m.zonotope(m.tensor(c), m.tensor(G))
"""
import numpy as np

from . import Interval, LinearSys, Tensor, Zonotope


def tensor(data):
    """tensor - a Tensor of the current backend from nested lists or a numpy array."""
    return Tensor(data)


def zonotope(c, G):
    """zonotope - the zonotope with center c (n) and generators G (n, m).

    MATLAB hands a scalar or a row for n = 1 or m = 1; the shapes are restored here.
    """
    c = c.reshape(-1)
    return Zonotope(c, G.reshape(c.shape[-1], -1))


def interval(inf, sup):
    """interval - the box inf <= x <= sup of vectors inf and sup."""
    return Interval(inf, sup)


def intervalMatrix(inf, sup):
    """intervalMatrix - the matrix of intervals inf <= X <= sup."""
    return Interval.matrix(inf, sup)


class ReachSet:
    """ReachSet - the reachable sets under CORA's field names: timeInt and timePoint."""

    def __init__(self, R):
        self.result = R
        self.timeInt = R.time_interval
        self.timePoint = R.time_point


class _LinearSys:
    """The linear system with CORA's positional reach and outputSet."""

    def __init__(self, sys):
        self.sys = sys

    def reach(self, X0, timeStep, tFinal, taylorTerms=10, linAlg="standard", U=None,
              zonotopeOrder=50):
        """reach - the reachable sets, as CORA's reach(sys, params, options) with these options."""
        names = {"wrapping-free": "wrapping_free", "wrappingFree": "wrapping_free"}
        alg = names.get(linAlg, linAlg)
        return ReachSet(self.sys.reach(X0, input_set=U, t_final=float(tFinal), alg=alg,
                                       time_step=float(timeStep), taylor_terms=int(taylorTerms),
                                       zonotope_order=int(zonotopeOrder)))

    def outputSet(self, R):
        """outputSet - the output sets y = C x of the reachable sets R."""
        return ReachSet(self.sys.output_set(R.result))


def linearSys(A, B=None, C=None):
    """linearSys - the linear system x' = A x, or x' = A x + B u, y = C x with B (and C)."""
    return _LinearSys(LinearSys(A) if B is None else LinearSys(A, B, C))


def zonotopes(sets):
    """zonotopes - (centers (N, n), generators (n, sum m) side by side, counts (N,)) of N zonotopes.

    One call instead of a few per set: MATLAB pays for every crossing into python.
    """
    centers = np.stack([np.asarray(Z.center) for Z in sets])
    generators = np.concatenate([np.asarray(Z.generators) for Z in sets], axis=1)
    counts = np.array([np.asarray(Z.generators).shape[1] for Z in sets], dtype=float)
    return centers, generators, counts
