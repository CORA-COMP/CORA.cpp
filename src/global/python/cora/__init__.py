"""cora - CORA.cpp: set-based computing on Eigen (numpy), or on libtorch (torch) if installed.

Sets are immutable values, stored batch first as PyTorch's tensors: Z.center (..., n), Z.generators
(..., m, n) one per row, Z.dim; operators +, -, &, |, in, M @ Z and Z @ M (linear maps) and s * Z (a
number only). Settings are scoped: `with cora.backend("torch"):`, `with cora.checks(False):`, `with
cora.strict_gradients():`, `with cora.tolerance(1e-10):` (the `tol` of every call that is given
none; default: `cora.default_tol(dtype)`). cora.tensor_ops() lists the Tensor operations with their
tags. cora.set_num_threads(n) sets the width of the thread pools (default: the CPU quota, the
affinity mask and the physical cores; env CORACPP_NUM_THREADS or OMP_NUM_THREADS). Errors are
cora.errors classes (cora.errors.DimensionMismatch, ...), warnings cora.CORAWarning. A Python
subclass of cora.ContSet, a pair registered with cora.register_binary and an algorithm registered
with cora.register_algorithm work like the built-in ones. Tensor(data, dtype) and
Tensor.zeros/ones/eye/randn build arrays on the current backend; Tensor.sin/cos/exp/... apply a
function to every element (no math or numpy needed).

Syntax:   X0 = cora.Zonotope(c, G)             c (..., n), G (..., m, n): one set or a batch
          sys = cora.LinearSys(A)              A (..., n, n): one system or a batch
          sys = cora.LinearSys(A, B, C)        x' = A x + B u, y = C x
          R = sys.reach(initial_set=X0, t_final=2.0, alg="standard", time_step=0.1)
          vdp = cora.NonlinearSys(lambda x: [x[1], (1 - x[0]**2) * x[1] - x[0]], 2)
          net = cora.NeuralNetwork([(W1, b1), (W2, b2)])      Y = net.evaluate(X): X's outputs
          cora.plot(R)                         or R.plot(); cora.subplots(2, 2).to_tikz(...)

Authors:       Claude Sonnet 5.5, Claude Opus 5.5
Written:       01-October-2026
Last update:   21-November-2026
Reviewed:      ---
"""
try:
    import torch  # noqa: F401  registers the tensor type that the compiled module returns
except ImportError:  # numpy and Eigen only
    pass

from . import banner, errors
from ._cora import (
    Capsule,
    ContSet,
    ConZonotope,
    DaeReachSet,
    Ellipsoid,
    EmptySet,
    Expr,
    Falsification,
    Fullspace,
    HybridAutomaton,
    HybridReachSet,
    Interval,
    LevelSet,
    LinearReset,
    LinearSys,
    LinParamSys,
    Location,
    LocationReachSet,
    NeuralNetwork,
    NeurNetContrSys,
    NnVerifyOptions,
    NonlinDASys,
    NonlinearSys,
    Polytope,
    PolyZonotope,
    ReachProblem,
    ReachSet,
    Rng,
    Specification,
    Taylm,
    Transition,
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
    cora_vnncomp_options,
    current_backend,
    current_tolerance,
    generate_random,
    has_torch,
    intersect,
    is_intersecting,
    isequal,
    lean,
    mat_exp,
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
    verify_strategies,
    vnncomp_options,
)
from .errors import CORAError, CORAWarning
from .extend import register_algorithm
from .figure import Figure, subplots, to_tikz
from .plot import plot, plot_over_time, plot_rand_point
from .tensor import Tensor, cos, default_tol, exp, eye, log, ones, randn, sin, sqrt, tan, zeros


# -----------------------------------------  BEGIN CODE  ----------------------------------------- #

# An example prints its CORA START block here and its CORA END block on exit.
banner.install(current_backend)

__all__ = [
    "CORAError", "CORAWarning", "Capsule", "ConZonotope", "ContSet", "DaeReachSet",
    "Ellipsoid", "EmptySet", "Expr",
    "Falsification", "Figure", "Fullspace", "HybridAutomaton", "HybridReachSet", "Interval",
    "LevelSet", "LinParamSys", "LinearReset", "LinearSys", "Location",
    "LocationReachSet",
    "NeuralNetwork", "NeurNetContrSys", "NnVerifyOptions", "NonlinDASys", "NonlinearSys",
    "PolyZonotope", "Polytope",
    "ReachProblem",
    "ReachSet", "Rng", "Specification", "Taylm", "Transition", "VerifyResult", "ZonoBundle",
    "Zonotope", "backend",
    "binary_pairs", "cart_prod", "checks",
    "checks_enabled", "contains", "convex_hull", "cora_color", "current_backend",
    "current_tolerance", "default_tol", "errors", "generate_random",
    "has_torch", "intersect", "is_intersecting", "isequal", "lean", "mat_exp", "num_threads",
    "plot",
    "plot_over_time", "plot_rand_point", "plus", "register_algorithm", "register_binary",
    "set_checks_enabled",
    "set_default_backend", "set_default_tolerance", "set_num_threads", "strict_gradients",
    "subplots", "tensor_ops",
    "to_tikz", "tolerance", "unite",
    "use_cora_colors", "verify_strategies", "vnncomp_options", "cora_vnncomp_options",
    "Tensor", "cos",
    "exp", "eye", "log", "ones", "randn", "sin", "sqrt", "tan", "zeros",
]

# ----------------------------------------  END OF CODE  ----------------------------------------- #
