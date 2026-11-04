"""CORA.cpp: set-based computing on Eigen (numpy), or on libtorch (torch) if installed.

    import cora
    X0 = cora.Zonotope(c, G)              # c (..., n), G (..., n, m): one set or a batch
    sys = cora.LinearSys(A)               # A (..., n, n): one system or a batch
    sys = cora.LinearSys(A, B, C)         # x' = A x + B u, y = C x
    R = sys.reach(initial_set=X0, t_final=2.0, alg="standard", time_step=0.1)
    vdp = cora.NonlinearSys(lambda x: [x[1], (1 - x[0]**2) * x[1] - x[0]], 2)
    net = cora.NeuralNetwork([(W1, b1), (W2, b2)])   # Y = net.evaluate(X) encloses the outputs of X
    cora.plot(R)                           # or R.plot(); cora.subplots(2, 2).to_tikz(...)

Sets are immutable values with properties (Z.center, Z.generators, Z.dim) and operators (+, @, *,
&, |, in, -). Settings are scoped: `with cora.backend("torch"):`, `with cora.checks(False):`,
`with cora.strict_gradients():`, `with cora.tolerance(1e-10):` (the `tol` of every call that
is given none; default: `cora.default_tol(dtype)`). cora.tensor_ops() lists the Tensor operations
with their tags. cora.set_num_threads(n) sets the width of the thread pools (default: the CPU
quota, the affinity mask and the physical cores; env CORACPP_NUM_THREADS or OMP_NUM_THREADS).
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
from ._cora import (
    ContSet,
    ConZonotope,
    EmptySet,
    ExpmMixedResult,
    ExpmResult,
    Expr,
    Falsification,
    Fullspace,
    Interval,
    IntervalMatrix,
    LinearSys,
    MatZonotope,
    MatZonotopeExpmMixedResult,
    MatZonotopeExpmOneParamResult,
    NeuralNetwork,
    NonlinearSys,
    Polytope,
    ReachProblem,
    ReachSet,
    Rng,
    Specification,
    VerifyResult,
    ZonoBundle,
    Zonotope,
    backend,
    binary_pairs,
    cart_prod,
    checks,
    checks_enabled,
    contains,
    convex_hull,
    cora_color,
    current_backend,
    current_tolerance,
    generate_random,
    has_torch,
    intersect,
    is_intersecting,
    isequal,
    lean,
    num_threads,
    plus,
    register_binary,
    set_checks_enabled,
    set_default_backend,
    set_default_tolerance,
    set_num_threads,
    strict_gradients,
    tensor_ops,
    tolerance,
    unite,
    use_cora_colors,
)
from .errors import CORAError, CORAWarning
from .extend import register_algorithm
from .figure import Figure, subplots, to_tikz
from .plot import plot, plot_over_time, plot_rand_point
from .tensor import Tensor, cos, default_tol, exp, eye, log, ones, randn, sin, sqrt, tan, zeros

# An example prints its CORA START block here and its CORA END block on exit.
banner.install(current_backend)

__all__ = [
    "CORAError", "CORAWarning", "ConZonotope", "ContSet", "EmptySet", "Expr", "ExpmMixedResult",
    "ExpmResult",
    "Falsification", "Figure", "Fullspace", "Interval", "IntervalMatrix", "LinearSys",
    "MatZonotope", "MatZonotopeExpmMixedResult", "MatZonotopeExpmOneParamResult",
    "NeuralNetwork", "NonlinearSys", "Polytope", "ReachProblem", "ReachSet", "Rng",
    "Specification", "VerifyResult", "ZonoBundle", "Zonotope", "backend", "binary_pairs",
    "cart_prod", "checks",
    "checks_enabled", "contains", "convex_hull", "cora_color", "current_backend",
    "current_tolerance", "default_tol", "errors", "generate_random",
    "has_torch", "intersect", "is_intersecting", "isequal", "lean", "num_threads", "plot",
    "plot_over_time", "plot_rand_point", "plus", "register_algorithm", "register_binary",
    "set_checks_enabled",
    "set_default_backend", "set_default_tolerance", "set_num_threads", "strict_gradients",
    "subplots", "tensor_ops",
    "to_tikz", "tolerance", "unite",
    "use_cora_colors", "Tensor", "cos",
    "exp", "eye", "log", "ones", "randn", "sin", "sqrt", "tan", "zeros",
]
