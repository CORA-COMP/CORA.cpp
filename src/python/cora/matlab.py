"""matlab - constructors for MATLAB CORA (toPy, fromPy), which cannot call pybind11 classes

MATLAB cannot take a pybind11 class itself (cora.Zonotope) but works with its instances, so these
functions build the objects. Data comes as python lists or numpy arrays.

    m = py.importlib.import_module('cora.matlab')
    Zpy = m.zonotope(m.tensor(c), m.tensor(G))
"""
import numpy as np

from . import Interval, LinearSys, Tensor, Zonotope


def tensor(data):
    """tensor - a Tensor of the current backend from nested lists or a numpy array."""
    return Tensor(data)


def zonotope(c, G):
    """zonotope - the zonotope with center c (n) and generators G (n, m)."""
    return Zonotope(c, G)


def interval(inf, sup):
    """interval - the box inf <= x <= sup of vectors inf and sup."""
    return Interval(inf, sup)


def intervalMatrix(inf, sup):
    """intervalMatrix - the matrix of intervals inf <= X <= sup."""
    return Interval.matrix(inf, sup)


def linearSys(A):
    """linearSys - the linear system x' = A x."""
    return LinearSys(A)


def zonotopes(sets):
    """zonotopes - (centers (N, n), generators (n, sum m) side by side, counts (N,)) of N zonotopes.

    One call instead of a few per set: MATLAB pays for every crossing into python.
    """
    centers = np.stack([np.asarray(Z.c) for Z in sets])
    generators = np.concatenate([np.asarray(Z.G) for Z in sets], axis=1)
    counts = np.array([np.asarray(Z.G).shape[1] for Z in sets], dtype=float)
    return centers, generators, counts
