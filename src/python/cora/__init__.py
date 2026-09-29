"""CORA.cpp: set-based computing on Eigen (numpy) or libtorch (torch), with CORA's names.

    import cora
    X0 = cora.Zonotope(c, G)              # c (..., n), G (..., n, m): one set or a batch
    sys = cora.LinearSys(A)               # A (..., n, n): one system or a batch
    R = sys.reach(X0, timeStep=0.1, tFinal=2.0)
    cora.plot(R)

Arrays: cora.Tensor(data, dtype), cora.zeros/ones/eye/randn build them on the current backend.

The classes are the C++ classes with the same methods; a batch lives in the object.
"""
from . import banner
from ._cora import (ContSet, Interval, LinearSys, Reach, Rng, Specification, Zonotope, backend,
                       setBackend)
from .colors import CORAcolor
from .tensor import Tensor, eye, ones, randn, zeros
from .plot import (plot, plot_initial_set, plot_interval, plot_points, plot_reach,
                   plot_simulation, plot_specification, plot_zonotope, zonotope_vertices)

# An example prints its CORA START block here and its CORA END block on exit.
banner.install(backend)

__all__ = [
    "CORAcolor", "ContSet", "Interval", "LinearSys", "Reach", "Rng", "Specification", "Zonotope",
    "backend", "plot", "plot_initial_set", "plot_interval", "plot_points", "plot_reach",
    "plot_simulation", "plot_specification", "plot_zonotope", "setBackend", "zonotope_vertices",
    "Tensor", "eye", "ones", "randn", "zeros",
]
