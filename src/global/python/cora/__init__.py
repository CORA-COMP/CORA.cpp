"""CORA.cpp: set-based computing on Eigen (numpy), or on libtorch (torch) if installed.

    import cora
    X0 = cora.Zonotope(c, G)              # c (..., n), G (..., n, m): one set or a batch
    sys = cora.LinearSys(A)               # A (..., n, n): one system or a batch
    sys = cora.LinearSys(A, B, C)         # x' = A x + B u, y = C x
    R = sys.reach(initial_set=X0, t_final=2.0, alg="standard", time_step=0.1)
    vdp = cora.NonlinearSys(lambda x: [x[1], (1 - x[0]**2) * x[1] - x[0]], 2)
    net = cora.NeuralNetwork([(W1, b1), (W2, b2)])   # Y = net.evaluate(X) encloses the outputs of X
    cora.plot(R)

Sets are immutable values with properties (Z.center, Z.generators, Z.dim) and operators (+, @, *,
&, |, in, -). Settings are scoped: `with cora.backend("torch"):`, `with cora.checks(False):`.
Errors are cora.errors classes (cora.errors.DimensionMismatch, ...), warnings cora.CORAWarning.
A Python subclass of cora.ContSet, a pair registered with cora.register_binary and an algorithm
registered with cora.register_algorithm work like the built-in ones.

Arrays: Tensor(data, dtype) and Tensor.zeros/ones/eye/randn build them on the current backend;
Tensor.sin/cos/exp/... apply a function to every element (no math or numpy needed).
"""
try:
    import torch  # noqa: F401  registers the tensor type that the compiled module returns
except ImportError:  # numpy and Eigen only
    pass

from . import banner, errors
from ._cora import (ContSet, Expr, Falsification, Interval, LinearSys, NeuralNetwork, NonlinearSys,
                    ReachProblem, ReachSet, Rng, Specification, VerifyResult, Zonotope, backend,
                    binary_pairs, cart_prod, checks, checks_enabled, contains, convex_hull,
                    cora_color, current_backend, has_torch, intersect, is_intersecting, lean, plus,
                    register_binary, set_checks_enabled, set_default_backend, unite,
                    use_cora_colors)
from .errors import CORAError, CORAWarning
from .extend import register_algorithm
from .tensor import Tensor, cos, exp, eye, log, ones, randn, sin, sqrt, tan, zeros
from .plot import plot

# An example prints its CORA START block here and its CORA END block on exit.
banner.install(current_backend)

__all__ = [
    "CORAError", "CORAWarning", "ContSet", "Expr", "Falsification", "Interval", "LinearSys",
    "NeuralNetwork", "NonlinearSys", "ReachProblem", "ReachSet", "Rng", "Specification",
    "VerifyResult", "Zonotope", "backend", "binary_pairs", "cart_prod", "checks", "checks_enabled",
    "contains", "convex_hull", "cora_color", "current_backend", "errors", "has_torch", "intersect",
    "is_intersecting", "lean", "plot", "plus", "register_algorithm", "register_binary",
    "set_checks_enabled", "set_default_backend", "unite", "use_cora_colors",
    "Tensor", "cos", "exp", "eye", "log", "ones", "randn", "sin", "sqrt", "tan", "zeros",
]
