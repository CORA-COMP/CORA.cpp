"""CORA.cpp: set-based computing on Eigen (numpy) or libtorch (torch), with CORA's names.

    import cora
    X0 = cora.Zonotope(c, G)              # c (..., n), G (..., n, m): one set or a batch
    sys = cora.LinearSys(A)               # A (..., n, n): one system or a batch
    vdp = cora.NonlinearSys(lambda x: [x[1], (1 - x[0]**2) * x[1] - x[0]], 2)
    net = cora.NeuralNetwork([(W1, b1), (W2, b2)])   # Y = net.evaluate(X) encloses the outputs of X
    R = sys.reach(X0, timeStep=0.1, tFinal=2.0)
    cora.plot(R)

Arrays: Tensor(data, dtype) and Tensor.zeros/ones/eye/randn build them on the current backend;
Tensor.sin/cos/exp/... apply a function to every element (no math or numpy needed).

The classes are the C++ classes with the same methods; a batch lives in the object.
"""
from . import banner
from ._cora import (CORAcolor, ContSet, Expr, Interval, LinearSys, NeuralNetwork, NonlinearSys,
                    Reach, Rng, Specification, Zonotope, backend, setBackend, useCORAcolors)
from .tensor import Tensor, cos, exp, eye, log, ones, randn, sin, sqrt, tan, zeros
from .plot import plot

# An example prints its CORA START block here and its CORA END block on exit.
banner.install(backend)

__all__ = [
    "CORAcolor", "ContSet", "Expr", "Interval", "LinearSys", "NeuralNetwork", "NonlinearSys",
    "Reach", "Rng", "Specification", "Zonotope", "backend", "plot", "setBackend", "useCORAcolors",
    "Tensor", "cos", "exp", "eye", "log", "ones", "randn", "sin", "sqrt", "tan", "zeros",
]
