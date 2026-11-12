// bindings - the compiled half of the `cora` Python package (`_cora`)
//
// The classes are the C++ classes, with the same names, so code reads the same in both languages
// and a batch lives in the object, not in the call. Python gets properties for what a set is
// (Z.center, Z.generators, Z.dim), operators (+, @, *, &, |, in, unary -), and keyword arguments
// for a problem and its algorithm. A Python subclass of cora.ContSet implements _dim, _center,
// _support_function, ... and then works with the checked entry points, the binary registry
// (register_binary) and plotting; an algorithm registered from Python (cora.register_algorithm)
// gets the problem checks of the built-in ones. Errors arrive as cora.errors classes, warnings
// through warnings.warn. The module holds Zonotope, Interval, EmptySet, ContSet, LinearSys,
// NonlinearSys, NeuralNetwork, Expr, ReachSet, Specification, VerifyResult, Rng, backend, checks,
// plus and register_binary.
//
// Syntax:   X0 = cora.Zonotope(c, G);   c (..., n), G (..., n, m): one set or a batch
//           sys = cora.LinearSys(A);   A (..., n, n): one system or a batch
//           R = sys.reach(initial_set=X0, t_final=2.0, alg="standard", time_step=0.1);
// See also: convert.h, the C++ headers, which document every method
//
// CORA.cpp only: the Python package has no CORA counterpart.
// Authors:       Claude Sonnet 5.5
// Written:       01-October-2026
// Last update:   11-November-2026
// Reviewed:      ---

#include "global/python/convert.h"

#include "cora/cont_dynamics/lin_param_sys/lin_param_sys.h"
#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/cont_dynamics/neur_net_contr_sys/neur_net_contr_sys.h"
#include "cora/cont_dynamics/nonlin_das_sys/nonlin_das_sys.h"
#include "cora/cont_dynamics/nonlinear_sys/nonlinear_sys.h"
#include "cora/cont_set/level_set/level_set.h"
#include "cora/cont_set/registry.h"
#include "cora/cont_set/taylm/taylm.h"
#include "cora/global/oracle/empty_set.h"
#include "cora/global/oracle/interval_matrix.h"
#include "cora/global/oracle/linear_sys.h"
#include "cora/global/oracle/oracle.h"
#include "cora/global/oracle/request.h"
#include "cora/global/plot/plot.h"
#include "cora/global/rng.h"
#include "cora/global/threads.h"
#include "cora/hybrid_dynamics/hybrid_automaton/hybrid_automaton.h"
#include "cora/matrix_set/interval_matrix/interval_matrix.h"
#include "cora/matrix_set/mat_zonotope/mat_zonotope.h"
#include "cora/nn/neural_network/neural_network.h"
#include "cora/specification/specification.h"

#include <limits>


using namespace cora;
using namespace cora::python;

// NOLINTBEGIN(backend,clones): numpy and torch arrays are converted to Tensor; .def rows repeat

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Errors, warnings and settings -------------------------------------------------------------------

/// cora::Error becomes the class of its identifier in cora.errors (a ValueError, ...).
void bind_errors(py::module_ &) {
    // NOLINTNEXTLINE(performance-unnecessary-value-param): pybind11 takes a function pointer
    py::register_exception_translator([](std::exception_ptr p) {
        try {
            if (p) std::rethrow_exception(p);
        } catch (const Error &e) {
            const py::object cls =
                py::module_::import("cora.errors").attr("BY_NAME")[Error::name(e.id())];
            PyErr_SetString(cls.ptr(), e.what());
        }
    });
    // warnings go to Python's warnings module, with the category cora.CORAWarning
    set_warning_handler([](const std::string &identifier, const std::string &message) {
        if (!Py_IsInitialized()) return;
        py::gil_scoped_acquire gil;
        const py::object category = py::module_::import("cora.errors").attr("CORAWarning");
        py::module_::import("warnings").attr("warn")(identifier + ": " + message, category, 2);
    });
}

/// A C++ scope as a Python context manager: entered by `with`, left at its end.
template <class Scope, class Arg>
struct PyScope {
    Arg arg;
    std::optional<Scope> scope;
};

template <class Scope, class Arg>
void bind_scope(py::module_ &m, const char *name, const char *doc) {
    using S = PyScope<Scope, Arg>;
    py::class_<S>(m, name, doc)
        .def(py::init([](Arg arg) { return new S{std::move(arg), std::nullopt}; }))
        .def(
            "__enter__",
            [](S &s) -> S & {
                s.scope.emplace(s.arg);
                return s;
            },
            py::return_value_policy::reference)
        .def("__exit__", [](S &s, const py::args &) { s.scope.reset(); });
}

void bind_settings(py::module_ &m) {
    // the scopes of the settings
    using Checks = PyScope<ChecksScope, bool>;
    using Backend = PyScope<BackendScope, std::string>;
    bind_scope<ChecksScope, bool>(m, "_ChecksScope", "The checks on or off in a with block");
    bind_scope<BackendScope, std::string>(m, "_BackendScope", "A backend in a with block");
    m.def(
        "checks", [](bool enabled) { return new Checks{enabled, std::nullopt}; },
        py::arg("enabled"), "with cora.checks(False): turns the input checks off in the block");
    m.def("set_checks_enabled", &set_checks_enabled, py::arg("enabled"),
          "The process default of the input checks (CORACPP_CHECKS=0 in the environment: off)");
    m.def("checks_enabled", &checks_enabled, "Whether this thread checks inputs");
    using Tolerance = PyScope<ToleranceScope, double>;
    bind_scope<ToleranceScope, double>(m, "_ToleranceScope", "A tolerance in a with block");
    m.def(
        "tolerance", [](double tol) { return new Tolerance{tol, std::nullopt}; }, py::arg("tol"),
        "with cora.tolerance(1e-10): the tol of every operation in the block that is given none");
    m.def("set_default_tolerance", &set_default_tolerance, py::arg("tol"),
          "The tolerance of threads without a cora.tolerance block (None: default_tol(dtype))");
    m.def("current_tolerance", &current_tolerance,
          "The tolerance set by a block or set_default_tolerance on this thread, else None");
    // the default tolerance of a dtype
    m.def(
        "_default_tol",
        [](const std::string &dtype) {
            return default_tol(dtype == "float32" ? DType::float32 : DType::float64);
        },
        py::arg("dtype"), "The tolerance of a dtype ('float64' or 'float32') without any override");
    m.def("set_num_threads", &set_num_threads, py::arg("n"),
          "The width of every thread pool (OpenMP, Eigen, torch); n < 1: the default (the least of "
          "the CPU quota, the affinity mask and the physical cores; CORACPP_NUM_THREADS and "
          "OMP_NUM_THREADS override it)");
    m.def("num_threads", &max_threads, "The width of the thread pools");
    m.def(
        "backend",
        [](const std::string &spec) {
#ifndef CORACPP_PYTHON_TORCH
            if (spec.rfind("torch", 0) == 0)
                CORA_FAIL(Error::install, "cora: this build of the Python package has no torch; "
                                          "use cora.backend('eigen')");
#endif
            // a new backend scope
            return new Backend{spec, std::nullopt};
        },
        py::arg("spec"),
        "with cora.backend('torch'): new arrays on 'eigen', 'torch', 'torch:cuda' in the block");
    m.def("set_default_backend", &set_default_backend, py::arg("spec"),
          "The backend of threads without a cora.backend block");
    m.def("current_backend", [] { return backend().name(); }, "The name of this thread's backend");
    using Strict = PyScope<GradientsScope, bool>;
    bind_scope<GradientsScope, bool>(m, "_GradientsScope", "Strict gradients in a with block");
    m.def(
        "strict_gradients", [](bool strict) { return new Strict{strict, std::nullopt}; },
        py::arg("strict") = true,
        "with cora.strict_gradients(): a not differentiable operation given tensors that need "
        "gradients is an error in the block, not a warning");
    m.def(
        "tensor_ops",
        [] {
            static const char *kinds[] = {"core", "derived", "fused"};
            static const char *grads[] = {"differentiable", "piecewise", "not_differentiable"};
            static const char *agree[] = {"exact", "ulp", "sum", "norm"};
            // the tensor operations with their kind and agreement
            py::list out;
            for (const OpInfo &op : tensor_ops())
                out.append(py::dict(py::arg("name") = op.name,
                                    py::arg("kind") = kinds[static_cast<int>(op.kind)],
                                    py::arg("gradient") = grads[static_cast<int>(op.gradient)],
                                    py::arg("agreement") = agree[static_cast<int>(op.agreement)],
                                    py::arg("k") = op.k, py::arg("summary") = op.summary));
            return out;
        },
        "The Tensor operations (ops.def): name, kind, gradient tag, agreement with lean, summary");
}

// Sets --------------------------------------------------------------------------------------------

/// A Python subclass of ContSet: the _impl layer calls the methods _dim, _center, ... of the
/// Python object.
class PyContSet : public ContSet {
  public:
    PyContSet() : ContSet(ClassId::custom) {}

  protected:
    std::string custom_class_name() const override {
        py::gil_scoped_acquire gil;
        return py::str(py::type::of(self()).attr("__name__"));
    }
    int64_t dim_impl() const override { return py::cast<int64_t>(call("_dim")); }
    Tensor center_impl() const override { return tensor_of(call("_center"), "_center"); }
    Tensor support_function_impl(const Tensor &d) const override {
        py::gil_scoped_acquire gil;
        const py::object value = call("_support_function", vector(d));
        if (py::isinstance<py::float_>(value) || py::isinstance<py::int_>(value))
            return Tensor::like(d, {py::cast<double>(value)}, {1, 1});
        return tensor_of(value, "_support_function");
    }
    Interval interval_impl() const override { return py::cast<Interval>(call("_interval")); }
    Tensor rand_point_impl(int64_t N, Rng &rng) const override {
        py::gil_scoped_acquire gil;
        // a random point of a set implemented in Python
        return tensor_of(call("_rand_point", N, py::cast(&rng, py::return_value_policy::reference)),
                         "_rand_point");
    }
    Set project_impl(const std::vector<int64_t> &dims) const override {
        py::gil_scoped_acquire gil;
        return set_of(call("_project", dims));
    }
    std::vector<Polygon> vertices_impl() const override {
        py::gil_scoped_acquire gil;
        std::vector<Polygon> out;
        for (const py::handle &p : call("_vertices")) {
            const auto a = py::array_t < double,
                       py::array::c_style | py::array::forcecast > ::ensure(p);
            Polygon polygon;
            for (py::ssize_t i = 0; i < a.shape(0); ++i)
                polygon.push_back({a.at(i, 0), a.at(i, 1)});
            out.push_back(polygon);
        }
        return out;
    }
    // the 3-D vertices come from the Python subclass, one array per face
    std::vector<std::vector<Point3>> vertices_3d_impl() const override {
        py::gil_scoped_acquire gil;
        std::vector<std::vector<Point3>> out;
        for (const py::handle &p : call("_vertices_3d")) {
            const auto a = py::array_t < double,
                       py::array::c_style | py::array::forcecast > ::ensure(p);
            std::vector<Point3> corners;
            corners.reserve(static_cast<std::size_t>(a.shape(0)));
            for (py::ssize_t i = 0; i < a.shape(0); ++i)
                corners.push_back({a.at(i, 0), a.at(i, 1), a.at(i, 2)});
            out.push_back(corners);
        }
        return out;
    }
    std::string display_impl() const override {
        py::gil_scoped_acquire gil;
        const py::function f = py::get_override(static_cast<const ContSet *>(this), "_display");
        if (f) return py::cast<std::string>(f());
        // the default display
        return class_name() + " of dimension " + std::to_string(dim());
    }

  private:
    /// The Python object of this set.
    py::handle self() const {
        return py::detail::get_object_handle(static_cast<const ContSet *>(this),
                                             py::detail::get_type_info(typeid(ContSet)));
    }
    /// The Python method `name`, which the subclass must define.
    template <class... A>
    py::object call(const char *name, A &&...args) const {
        py::gil_scoped_acquire gil;
        const py::function f = py::get_override(static_cast<const ContSet *>(this), name);
        if (!f)
            CORA_FAIL(Error::not_supported, class_name() + " does not define " + name +
                                                "; a subclass of cora.ContSet implements it");
        return f(std::forward<A>(args)...);
    }
};

/// A polygon as an (k, 2) array.
py::array_t<double> polygon_array(const Polygon &polygon) {
    py::array_t<double> out({static_cast<py::ssize_t>(polygon.size()), py::ssize_t(2)});
    auto view = out.mutable_unchecked<2>();
    for (std::size_t i = 0; i < polygon.size(); ++i) {
        view(i, 0) = polygon[i][0];
        view(i, 1) = polygon[i][1];
    }
    return out;
}

/// The points of space as an (k, 3) array.
py::array_t<double> points3_array(const std::vector<Point3> &points) {
    py::array_t<double> out({static_cast<py::ssize_t>(points.size()), py::ssize_t(3)});
    auto view = out.mutable_unchecked<2>();
    for (std::size_t i = 0; i < points.size(); ++i)
        for (py::ssize_t k = 0; k < 3; ++k) view(i, k) = points[i][static_cast<std::size_t>(k)];
    return out;
}

/// Whether o is a plain number.
bool is_number(const py::handle &o) {
    return py::isinstance<py::float_>(o) || py::isinstance<py::int_>(o);
}

/// An optional vector argument: None stays empty, a number is a 1 x 1 array (one value for all).
std::optional<Tensor> optional_vector(const py::object &o, const char *what) {
    if (o.is_none()) return std::nullopt;
    if (is_number(o)) return Tensor::from_data({py::cast<double>(o)}, {1, 1});
    return vector_of(o, what);
}

py::object not_implemented() { return py::reinterpret_borrow<py::object>(Py_NotImplemented); }

/// A binary operation of two Python sets through the registry, as the object of its class.
template <Set (*Op)(const ContSet &, const ContSet &)>
py::object binary(const py::object &S1, const py::object &S2) {
    return object_of(Op(set_of(S1), set_of(S2)));
}

/// The functions of the registry, and registering a pair from Python.
void bind_registry(py::module_ &m) {
    m.def("plus", &binary<&plus>, py::arg("S1"), py::arg("S2"),
          "The Minkowski sum of two sets of any classes");
    m.def("intersect", &binary<&intersect>, py::arg("S1"), py::arg("S2"),
          "The intersection (CORA's and)");
    m.def("unite", &binary<&unite>, py::arg("S1"), py::arg("S2"),
          "An enclosure of the union (CORA's or)");
    m.def("convex_hull", &binary<&convex_hull>, py::arg("S1"), py::arg("S2"),
          "An enclosure of the convex hull");
    m.def("cart_prod", &binary<&cart_prod>, py::arg("S1"), py::arg("S2"), "The Cartesian product");
    m.def(
        "contains",
        [](const py::object &S1, const py::object &S2, Tol tol) {
            return contains(set_of(S1), set_of(S2), tol);
        },
        py::arg("S1"), py::arg("S2"), py::arg("tol") = py::none(),
        "Whether S2 is a subset of S1 enlarged by tol (infinity norm); tol=None: the "
        "repository tolerance");
    // the intersection predicate of two sets
    m.def(
        "is_intersecting",
        [](const py::object &S1, const py::object &S2, Tol tol) {
            return is_intersecting(set_of(S1), set_of(S2), tol);
        },
        py::arg("S1"), py::arg("S2"), py::arg("tol") = py::none(),
        "Whether S1 and S2 intersect, up to tol (None: the repository tolerance)");
    m.def(
        "isequal",
        [](const py::object &S1, const py::object &S2, Tol tol) {
            return isequal(set_of(S1), set_of(S2), tol);
        },
        py::arg("S1"), py::arg("S2"), py::arg("tol") = py::none(),
        "Whether two sets are equal, up to tol (None: the repository tolerance)");
    m.def(
        "binary_pairs",
        [] {
            py::list out;
            for (const BinaryPair &p : binary_pairs())
                out.append(py::make_tuple(binary_op_name(p.op), p.left, p.right));
            // the pairs as (op, left, right) tuples
            return out;
        },
        "The pairs of classes each binary operation is implemented for: (op, left, right)");
    m.def(
        "register_binary",
        [](const std::string &op, const py::type &left, const py::type &right,
           const py::function &f) {
            // the function is kept alive, and freed with the GIL held
            auto fn = std::shared_ptr<py::object>(new py::object(f), [](py::object *o) {
                if (!python_alive()) {
                    o->release(); // the interpreter is gone (static destruction): leak it
                    delete o;
                    return;
                }
                py::gil_scoped_acquire gil;
                delete o;
            });
            const BinaryOp o = binary_op(op);
            const std::string l = py::str(left.attr("__name__")),
                              r = py::str(right.attr("__name__"));
            const auto ref = [](const ContSet &S) {
                return py::cast(&S, py::return_value_policy::reference);
            };
            if (o == BinaryOp::contains || o == BinaryOp::is_intersecting ||
                o == BinaryOp::isequal) {
                register_binary(
                    o, l, r,
                    PredicateFunction([fn, ref](const ContSet &S1, const ContSet &S2, Tol tol) {
                        py::gil_scoped_acquire gil;
                        // a Python function always gets a number: the repository tolerance
                        return py::cast<bool>((*fn)(ref(S1), ref(S2), resolve_tol(tol)));
                    }));
            } else {
                register_binary(o, l, r,
                                SetFunction([fn, ref](const ContSet &S1, const ContSet &S2) {
                                    py::gil_scoped_acquire gil;
                                    return set_of((*fn)(ref(S1), ref(S2)));
                                }));
            }
        },
        py::arg("op"), py::arg("left"), py::arg("right"), py::arg("f"),
        "Registers f(S1, S2) for op between the classes left and right (looked up after the "
        "built-in pairs)");
}

Tensor::Norm norm_of(const py::object &type);

/// What every set has; bound on the base class, so any set works wherever a set is expected,
/// including the operators between sets of any classes (through the registry).
void bind_cont_set(py::module_ &m) {
    m.def(
        "generate_random",
        [](cora::Rng &rng, std::optional<int64_t> dimension,
           const std::optional<std::vector<std::string>> &admissible) {
            return object_of(admissible ? ContSet::generate_random(rng, dimension, *admissible)
                                        : ContSet::generate_random(rng, dimension));
        },
        py::arg("rng"), py::arg("dimension") = py::none(), py::arg("admissible") = py::none(),
        "A random set of one of the admissible classes ('interval', 'zonotope'); the class's own "
        "generate_random takes its options");
    // the abstract set, to be subclassed in Python
    py::class_<ContSet, PyContSet>(m, "ContSet",
                                   "The abstract set; a Python subclass implements _dim, _center, "
                                   "_support_function, _interval, _rand_point, _project, "
                                   "_vertices, _vertices_3d")
        .def(py::init<>())
        .def_property_readonly("dim", &ContSet::dim, "The dimension n")
        .def_property_readonly(
            "center", [](const ContSet &S) { return bounds(S.center()); }, "The center (..., n)")
        .def_property_readonly("class_name", &ContSet::class_name,
                               "The name of the class: 'Zonotope', 'Interval', ...")
        .def(
            "support_function",
            [](const ContSet &S, const py::object &d) {
                return scalar(S.support_function(vector_of(d, "the direction")));
            },
            py::arg("d"), "max of d'x over the set; d (..., n)")
        .def("interval", &ContSet::interval, "The interval hull")
        .def(
            "rand_point",
            [](const ContSet &S, int64_t N, cora::Rng &rng) {
                // random points as columns
                return matrix(S.rand_point(N, rng));
            },
            py::arg("N"), py::arg("rng"), "N random points as columns (..., n, N)")
        .def(
            "project",
            [](const ContSet &S, const std::vector<int64_t> &dims) {
                return object_of(S.project(dims));
            },
            py::arg("dims"), "The projection onto the dimensions dims (0-based)")
        .def(
            "vertices",
            [](const ContSet &S) {
                py::list polygons;
                for (const Polygon &p : S.vertices()) polygons.append(polygon_array(p));
                return polygons;
            },
            "The vertices of a two-dimensional set, one (k, 2) array per batch member")
        .def(
            "vertices_3d",
            [](const ContSet &S) {
                // the corner lists of the faces
                py::list corners;
                for (const std::vector<Point3> &p : S.vertices_3d())
                    corners.append(points3_array(p));
                return corners;
            },
            "The vertices of a three-dimensional set, one (k, 3) array per batch member")
        .def("is_bounded", &ContSet::is_bounded, "Whether the set is bounded")
        .def("is_full_dim", &ContSet::is_full_dim, py::arg("tol") = py::none(),
             "Whether the set has a nonempty interior; tol=None: the repository tolerance")
        .def(
            "lift",
            [](const ContSet &S, int64_t N, const std::optional<std::vector<int64_t>> &proj) {
                return object_of(S.lift(N, proj));
            },
            py::arg("N"), py::arg("proj") = py::none(),
            "The set in N dimensions with coordinate j at position proj[j] (0-based; default "
            "0..dim-1)")
        .def(
            "norm",
            [](const ContSet &S, const py::object &type, const std::string &mode) {
                // the norm bound
                return scalar(S.norm(norm_of(type), mode));
            },
            py::arg("type") = 2, py::arg("mode") = "ub",
            "An upper bound of the norm (1, 2, inf or 'fro') of the points; -inf for the empty "
            "set")
        .def(
            "enclose", [](const ContSet &S, const ContSet &S2) { return object_of(S.enclose(S2)); },
            py::arg("S2"), "A set of the class of S that encloses S and S2")
        .def(
            "reduce", [](const ContSet &S, int order) { return object_of(S.reduce(order)); },
            py::arg("order") = 1, "A set with at most order * dim generators that encloses S")
        .def(
            "linear_map",
            [](const ContSet &S, const py::object &M) {
                return object_of(S.linear_map(tensor_of(M, "the matrix")));
            },
            py::arg("M"), "The image under the matrix M (..., m, n)")
        .def(
            "enlarge",
            [](const ContSet &S, const py::object &factor) {
                // enlarge by a number or per dimension
                if (is_number(factor)) return object_of(S.enlarge(py::cast<double>(factor)));
                return object_of(S.enlarge(vector_of(factor, "the factor")));
            },
            py::arg("factor"), "The set scaled about its center by a number or per dimension")
        .def(
            "minus",
            [](const ContSet &S, const py::object &p) {
                return object_of(S.minus(vector_of(p, "the vector")));
            },
            py::arg("p"), "The set shifted by -p")
        .def(
            "times",
            [](const ContSet &S, const py::object &factor) {
                if (is_number(factor)) return object_of(S.times(py::cast<double>(factor)));
                return object_of(S.times(vector_of(factor, "the factor")));
            },
            py::arg("factor"), "The set scaled by a number or per dimension, diag(factor) S")
        .def(
            "compact",
            [](const ContSet &S, const std::optional<std::string> &method, Tol tol) {
                // compact with an optional method
                return object_of(S.compact(method, tol));
            },
            py::arg("method") = py::none(), py::arg("tol") = py::none(),
            "The set without redundancy; a variant of the class (a zonotope: 'zeros', 'all', "
            "'aligned'); tol=None: the repository tolerance")
        .def(
            "__neg__", [](const ContSet &S) { return object_of(S.uminus()); }, "-S")
        .def(
            "__pos__", [](const ContSet &S) { return object_of(S.uplus()); }, "S itself")
        .def(
            "minkowski_difference",
            [](const ContSet &S, const py::object &S2, const std::optional<std::string> &type) {
                return object_of(S.minkowski_difference(set_of(S2), type));
            },
            py::arg("S2"), py::arg("type") = py::none(),
            "The Minkowski difference {x | x + S2 in S}")
        .def(
            "decompose",
            [](const ContSet &S, const std::vector<std::array<int64_t, 2>> &blocks) {
                py::list parts;
                // the parts of a decomposition
                for (const Set &part : S.decompose(blocks)) parts.append(object_of(part));
                return parts;
            },
            py::arg("blocks"),
            "The projections onto blocks [first, last] of coordinates (0-based, inclusive)")
        .def(
            "project_high_dim",
            [](const ContSet &S, int64_t N, const std::vector<int64_t> &proj) {
                return object_of(S.project_high_dim(N, proj));
            },
            py::arg("N"), py::arg("proj"),
            "The set in N dimensions, coordinate j placed at proj[j]")
        .def(
            "project_on_hyperplane",
            [](const ContSet &S, const py::object &c, const py::object &d) {
                const Tensor normal = vector_of(c, "the normal");
                const Tensor offset = is_number(d) ? normal.full_like({1, 1}, py::cast<double>(d))
                                                   : tensor_of(d, "the offset");
                // the projection onto a hyperplane
                return object_of(S.project_on_hyperplane(normal, offset));
            },
            py::arg("c"), py::arg("d"),
            "The orthogonal projection onto the hyperplane {x | c'x = d}")
        .def("__eq__",
             [](const py::object &S1, const py::object &S2) -> py::object {
                 if (!py::isinstance<ContSet>(S2)) return not_implemented();
                 return py::cast(eq(set_of(S1), set_of(S2)));
             })
        .def("__ne__",
             [](const py::object &S1, const py::object &S2) -> py::object {
                 if (!py::isinstance<ContSet>(S2)) return not_implemented();
                 return py::cast(ne(set_of(S1), set_of(S2)));
             })
        .def("display", &ContSet::display,
             "The text that `print(S)` shows, as CORA displays the set")
        .def("__repr__", &ContSet::display)
        .def("__str__", &ContSet::display)
        .def("__add__",
             [](const py::object &S1, const py::object &S2) -> py::object {
                 // operators with other sets
                 if (!py::isinstance<ContSet>(S2)) return not_implemented();
                 return binary<&plus>(S1, S2);
             })
        .def("__and__",
             [](const py::object &S1, const py::object &S2) -> py::object {
                 if (!py::isinstance<ContSet>(S2)) return not_implemented();
                 return binary<&intersect>(S1, S2);
             })
        .def("__or__",
             [](const py::object &S1, const py::object &S2) -> py::object {
                 if (!py::isinstance<ContSet>(S2)) return not_implemented();
                 return binary<&unite>(S1, S2);
             })
        .def("__contains__", [](const py::object &S1, const py::object &S2) {
            if (py::isinstance<ContSet>(S2)) return contains(set_of(S1), set_of(S2));
            if (py::isinstance<Interval>(S1))
                return py::cast<const Interval &>(S1).contains(vector_of(S2, "the point"));
            if (py::isinstance<Zonotope>(S1))
                return py::cast<const Zonotope &>(S1).contains(tensor_of(S2, "the points"));
            // contains dispatches on the class of the first set
            if (py::isinstance<ConZonotope>(S1))
                return py::cast<const ConZonotope &>(S1).contains(tensor_of(S2, "the points"));
            if (py::isinstance<ZonoBundle>(S1))
                return py::cast<const ZonoBundle &>(S1).contains(tensor_of(S2, "the points"));
            if (py::isinstance<Polytope>(S1))
                return py::cast<const Polytope &>(S1).contains(tensor_of(S2, "the points"));
            if (py::isinstance<Ellipsoid>(S1))
                return py::cast<const Ellipsoid &>(S1).contains(tensor_of(S2, "the points"));
            if (py::isinstance<Capsule>(S1))
                return py::cast<const Capsule &>(S1).contains(tensor_of(S2, "the points"));
            CORA_FAIL(Error::not_supported,
                      "point containment is implemented for intervals, zonotopes, constrained "
                      "zonotopes, zonotope bundles, polytopes, ellipsoids and capsules only");
        });
    bind_registry(m);
}

/// The matrices of a Python list of arrays, for quad_map.
std::vector<Tensor> matrix_list_of(const py::list &Q) {
    std::vector<Tensor> out;
    for (const py::handle &M : Q) out.push_back(tensor_of(M, "a matrix of Q"));
    return out;
}

/// A Python number or array as the operand of a box: a number is a 1 x 1 array, else a column.
Tensor operand(const Interval &I, const py::object &o, const char *what) {
    if (is_number(o)) return I.inf().full_like({1, 1}, py::cast<double>(o));
    return vector_of(o, what);
}

/// The norm of a Python value: 1, 2, inf or "fro".
Tensor::Norm norm_of(const py::object &type) {
    if (py::isinstance<py::str>(type)) {
        if (py::cast<std::string>(type) == "fro") return Tensor::Norm::fro;
    } else if (is_number(type)) {
        const double p = py::cast<double>(type);
        if (p == 1) return Tensor::Norm::one;
        if (p == 2) return Tensor::Norm::two;
        if (std::isinf(p) && p > 0) return Tensor::Norm::inf;
    }
    CORA_FAIL(Error::wrong_value, "cora: the norm type must be 1, 2, inf or 'fro'");
}

/// A set, or points as the interval [p, p], as the operand of a whole space.
Set operand_set(const py::object &o) {
    if (py::isinstance<ContSet>(o)) return set_of(o);
    const Tensor p = tensor_of(o, "the point");
    return Set(Interval(p, p));
}

void bind_fullspace(py::module_ &m) {
    py::class_<Fullspace, ContSet>(m, "Fullspace", "The whole space R^n.")
        .def(py::init<int64_t>(), py::arg("n"), "R^n, n >= 0")
        .def_static("inf", &Fullspace::inf, py::arg("n"), "R^n (CORA's fullspace.Inf)")
        .def("box", &Fullspace::box, "The smallest box around the set: the set itself")
        .def(
            "intersect",
            [](const Fullspace &F, const py::object &S) {
                return object_of(F.intersect(set_of(S)));
            },
            py::arg("S"), "The intersection with a set: that set")
        .def(
            "plus",
            [](const Fullspace &F, const py::object &S) {
                // the sum with the whole space
                return object_of(F.plus(operand_set(S).get()));
            },
            py::arg("S"),
            "The Minkowski sum with a set or a point: the whole space, the empty set for an empty "
            "S")
        .def(
            "convex_hull",
            [](const Fullspace &F, const py::object &S) { return F.convex_hull(set_of(S).get()); },
            py::arg("S"), "The convex hull with a set: the whole space")
        // the queries take points or boxes
        .def(
            "contains",
            [](const Fullspace &F, const py::object &S, Tol tol) {
                return F.contains(operand_set(S).get(), tol);
            },
            py::arg("S"), py::arg("tol") = py::none(), "Whether the set contains S: always")
        // the box of any set decides the intersection
        .def(
            "is_intersecting",
            [](const Fullspace &F, const py::object &S, Tol tol) {
                // the whole space meets every set unless it is empty
                return F.is_intersecting(operand_set(S).get(), tol);
            },
            py::arg("S"), py::arg("tol") = py::none(), "Whether the set meets S: unless S is empty")
        .def(
            "isequal",
            [](const Fullspace &F, const py::object &S, Tol tol) {
                return F.isequal(operand_set(S).get(), tol);
            },
            py::arg("S"), py::arg("tol") = py::none(),
            "Whether S is the whole space of the same dimension")
        .def("is_bounded", &Fullspace::is_bounded, "Whether the set is bounded: only in R^0")
        .def_static(
            "enclose_points",
            [](const py::object &points) {
                return Fullspace::enclose_points(tensor_of(points, "the points"));
            },
            py::arg("points"), "The whole space of the dimension of the points (n, N)")
        .def("project_high_dim", &Fullspace::project_high_dim, py::arg("N"), py::arg("proj"),
             "The whole space again, for N = dim; any other N is refused (CORA: not_defined)")
        // a random set of the whole space
        .def_static("generate_random", &Fullspace::generate_random, py::arg("rng"),
                    py::arg("dimension") = py::none(),
                    "The whole space of the dimension, or of a random one in 1..10")
        .def("radius", &Fullspace::radius, "The radius of the smallest ball around it: Inf")
        .def("volume", &Fullspace::volume, "The volume: Inf")
        .def("complement", &Fullspace::complement, "The complement: the empty set")
        .def("representsa", &Fullspace::representsa, py::arg("type"), py::arg("tol") = py::none(),
             "Whether the set is also a set of CORA's type: 'interval', 'convexSet', 'fullspace'")
        .def(
            "linear_map",
            [](const Fullspace &F, const py::object &M) {
                if (py::isinstance<Interval>(M))
                    return object_of(F.linear_map(py::cast<Interval>(M)));
                return object_of(F.linear_map(tensor_of(M, "the matrix")));
            },
            py::arg("M"), "The image under a matrix, an enclosure when M lacks full row rank");
}

/// The intervals of a list whose items are intervals or points (arrays, as point intervals).
std::vector<Interval> intervals_of(const py::list &items) {
    std::vector<Interval> Is;
    for (const py::handle &item : items) {
        if (py::isinstance<Interval>(item)) {
            Is.push_back(py::cast<Interval>(item));
        } else {
            const Tensor p = vector_of(item, "the number");
            Is.emplace_back(p, p);
        }
    }
    return Is;
}

void bind_interval(py::module_ &m) {
    py::class_<Interval, ContSet>(m, "Interval",
                                  "A box, or a matrix of intervals: inf <= x <= sup.")
        .def(py::init([](const Array &inf, const Array &sup) {
                 // a box from its bounds
                 return Interval(vector_of(inf, "inf"), vector_of(sup, "sup"));
             }),
             py::arg("inf"), py::arg("sup"), "The box of the vectors inf <= sup (..., n)")
        .def_static(
            "matrix",
            [](const Array &inf, const Array &sup) {
                return Interval(tensor_of(inf, "inf"), tensor_of(sup, "sup"));
            },
            py::arg("inf"), py::arg("sup"), "An interval matrix: bounds (..., n, k)")
        .def_static("stack", &Interval::stack, py::arg("intervals"),
                    "A batch of intervals as one; the batch is the leading dimension")
        .def_static(
            "generate_random",
            [](cora::Rng &rng, const std::optional<int64_t> &dimension, const py::object &center,
               const py::object &max_radius) {
                // a random box
                return Interval::generate_random(rng, dimension, optional_vector(center, "center"),
                                                 optional_vector(max_radius, "max_radius"));
            },
            py::arg("rng"), py::arg("dimension") = py::none(), py::arg("center") = py::none(),
            py::arg("max_radius") = py::none(),
            "A random box: the dimension (default that of center or max_radius, else random in "
            "1..10), a center and the maximum radii (a vector or one number)")
        .def_property_readonly(
            "inf", [](const Interval &I) { return bounds(I.inf()); }, "The lower bounds (..., n)")
        .def_property_readonly(
            "sup", [](const Interval &I) { return bounds(I.sup()); }, "The upper bounds (..., n)")
        .def_property_readonly(
            "rad", [](const Interval &I) { return bounds(I.rad()); },
            "The radii (sup - inf) / 2, (..., n)")
        .def(
            "linear_map",
            [](const Interval &I, const py::object &M) {
                return I.linear_map(tensor_of(M, "the matrix"));
            },
            py::arg("M"), "M I enclosed by its interval hull")
        // the box of all points
        .def("plus", &Interval::plus, py::arg("I2"), "The sum with an interval, as a box")
        .def_static("unbounded", &Interval::unbounded, py::arg("n"),
                    "The box (-Inf, Inf)^n (CORA: interval.Inf)")
        .def("abs", &Interval::abs, "The image under the elementwise absolute value")
        .def("__abs__", &Interval::abs)
        .def("intersect", &Interval::intersect, py::arg("I2"),
             "The intersection; bounds without columns if empty")
        .def(
            "cart_prod",
            [](const Interval &I, const py::object &other) {
                if (py::isinstance<Interval>(other)) return I.cart_prod(py::cast<Interval>(other));
                return I.cart_prod(vector_of(other, "the point"));
            },
            py::arg("other"), "The Cartesian product with an interval or a point")
        .def_static("cat", &Interval::cat, py::arg("dim"), py::arg("intervals"),
                    "The intervals joined along dim (1: stacked, 2: side by side)")
        .def(
            "boundary_point",
            [](const Interval &I, const py::object &dir, const py::object &start) {
                std::optional<Tensor> from;
                // the start point defaults to the center
                if (!start.is_none()) from = vector_of(start, "the start point");
                return bounds(I.boundary_point(vector_of(dir, "dir"), from));
            },
            py::arg("dir"), py::arg("start") = py::none(),
            "The point where the ray from start (default: the center) along dir leaves the box")
        .def("acos", &Interval::acos, "The image under the arccosine")
        .def("asin", &Interval::asin, "The image under the arcsine")
        .def("atan", &Interval::atan, "The image under the arctangent")
        .def("acosh", &Interval::acosh, "The image under the inverse hyperbolic cosine")
        .def("asinh", &Interval::asinh, "The image under the inverse hyperbolic sine")
        .def("atanh", &Interval::atanh, "The image under the inverse hyperbolic tangent")
        .def("atan2", &Interval::atan2, py::arg("I2"),
             "The hull of atan2(y, x) at the four corners, y from this interval, x from I2")
        .def("cos", &Interval::cos, "The image under the cosine (CORA's case analysis)")
        .def("cosh", &Interval::cosh, "The image under the hyperbolic cosine")
        .def("exp", &Interval::exp, "The image under the exponential")
        .def("convex_hull", &Interval::convex_hull, py::arg("I2"),
             "The hull of two boxes; an empty one returns the other")
        // hull of the points
        .def_static(
            "enclose_points",
            [](const py::object &points) {
                return Interval::enclose_points(tensor_of(points, "the points"));
            },
            py::arg("points"), "The smallest box around the points, the columns of (n, N)")
        .def_static("empty", &Interval::empty, py::arg("n"), "The empty interval of n dimensions")
        .def_static(
            "horzcat", [](const py::list &items) { return Interval::horzcat(intervals_of(items)); },
            py::arg("items"), "Intervals (or points, as arrays) joined side by side")
        .def_static(
            "vertcat", [](const py::list &items) { return Interval::vertcat(intervals_of(items)); },
            py::arg("items"), "Intervals (or points, as arrays) stacked")
        .def(
            "enlarge",
            [](const Interval &I, const py::object &factor) {
                if (is_number(factor)) return I.enlarge(py::cast<double>(factor));
                return I.enlarge(vector_of(factor, "the factor"));
            },
            py::arg("factor"), "The box scaled about its center by a number or per dimension")
        // transposes
        .def("ctranspose", &Interval::ctranspose, "The transpose of an interval matrix")
        .def("transpose", &Interval::transpose, "The transpose of an interval matrix")
        .def_property_readonly("T", &Interval::transpose, "The transposed box")
        .def(
            "supremum", [](const Interval &I) { return bounds(I.supremum()); }, "The upper bound")
        .def(
            "split", [](const Interval &I, int64_t n) { return I.split(n); }, py::arg("n"),
            "The two halves of the box cut at the center of coordinate n (0-based)")
        .def("sum", &Interval::sum, py::arg("dim") = py::none(),
             "The sum of the entries along dim (1: down the rows, 2: along the columns)")
        .def("tril", &Interval::tril, py::arg("K") = 0,
             "The lower triangular part from the K-th diagonal on")
        .def("triu", &Interval::triu, py::arg("K") = 0,
             "The upper triangular part from the K-th diagonal on")
        .def(
            "volume", [](const Interval &I) { return scalar(I.volume()); },
            "The product of the widths of the box")
        .def("zonotope", &Interval::zonotope,
             "The zonotope with the center and the radius as generators")
        .def("sqrt", &Interval::sqrt, "The image under the square root")
        // tangent functions
        .def("tan", &Interval::tan,
             "The image under the tangent (the line where a pole may lie inside)")
        .def("tanh", &Interval::tanh, "The image under the hyperbolic tangent")
        .def(
            "times",
            [](const Interval &I, const py::object &other) {
                if (py::isinstance<Interval>(other)) return I.times(py::cast<Interval>(other));
                return I.times(operand(I, other, "the factor"));
            },
            py::arg("other"), "The entrywise product with a box or with numbers (CORA's .*)")
        .def("diag", &Interval::diag, py::arg("k") = 0,
             "A vector as a diagonal matrix, a matrix as its k-th diagonal")
        .def("dirac", &Interval::dirac, py::arg("n") = 0,
             "The n-th derivative of the Dirac delta on the box")
        .def(
            "infimum", [](const Interval &I) { return bounds(I.infimum()); }, "The lower bound")
        .def("interval_matrix", &Interval::interval_matrix,
             "The interval matrix [c - r, c + r] of the center and the radius")
        .def("is_bounded", &Interval::is_bounded, "Whether no bound is infinite")
        // predicates of the box
        .def("is_full_dim", &Interval::is_full_dim, py::arg("tol") = py::none(),
             "Whether the box has a nonempty interior; tol=None: the repository tolerance")
        .def(
            "is_intersecting",
            [](const Interval &I, const py::object &other, Tol tol) {
                if (py::isinstance<Interval>(other))
                    return I.is_intersecting(py::cast<Interval>(other), tol);
                return I.is_intersecting(vector_of(other, "the point"), tol);
            },
            py::arg("other"), py::arg("tol") = py::none(),
            "Whether the box meets another box or contains a point, up to tol")
        .def(
            "isequal",
            [](const Interval &I, const py::object &other, Tol tol) {
                if (py::isinstance<Interval>(other))
                    return I.isequal(py::cast<Interval>(other), tol);
                return I.isequal(tensor_of(other, "the point"), tol);
            },
            py::arg("other"), py::arg("tol") = py::none(),
            "Whether the bounds agree up to tol; infinite bounds must be the same")
        // comparisons of boxes
        .def("less_equal", &Interval::less_equal, py::arg("I2"), "Whether the box lies in I2")
        .def("less", &Interval::less, py::arg("I2"), "Whether the box lies strictly in I2")
        .def(
            "minus",
            [](const Interval &I, const py::object &other) {
                if (py::isinstance<Interval>(other)) return I.minus(py::cast<Interval>(other));
                return I.minus(operand(I, other, "the subtrahend"));
            },
            py::arg("other"), "The difference with a box, a number or a column")
        .def("divide", &Interval::divide, py::arg("d"),
             "The box divided by a number (CORA's mrdivide); d = 0 gives the whole line")
        .def(
            "kron",
            [](const Interval &I, const py::object &other, bool left) {
                if (py::isinstance<Interval>(other)) return I.kron(py::cast<Interval>(other));
                const Tensor A = tensor_of(other, "the factor");
                return left ? Interval::kron(A, I) : I.kron(A);
            },
            py::arg("other"), py::arg("left") = false,
            "The Kronecker product with a box or a matrix; left=True puts the matrix first")
        // matrix powers and maxima
        .def("matrix_power", &Interval::matrix_power, py::arg("exponent"),
             "The power of a square interval matrix")
        .def(
            "max",
            [](const Interval &I, const py::object &other) -> py::object {
                if (other.is_none()) return bounds(I.max());
                if (py::isinstance<ContSet>(other))
                    return py::cast(I.max(py::cast<const ContSet &>(other)));
                return py::cast(I.max(operand(I, other, "the operand")));
            },
            py::arg("other") = py::none(),
            "The elementwise larger of two boxes; without an argument the upper bound")
        .def(
            "min",
            [](const Interval &I, const py::object &other) -> py::object {
                if (other.is_none()) return bounds(I.min());
                if (py::isinstance<ContSet>(other))
                    return py::cast(I.min(py::cast<const ContSet &>(other)));
                // minima of two boxes
                return py::cast(I.min(operand(I, other, "the operand")));
            },
            py::arg("other") = py::none(),
            "The elementwise smaller of two boxes; without an argument the lower bound")
        .def(
            "minkowski_difference",
            [](const Interval &I, const py::object &S, const std::optional<std::string> &type) {
                if (py::isinstance<Interval>(S))
                    return I.minkowski_difference(py::cast<Interval>(S));
                if (py::isinstance<ContSet>(S))
                    return I.minkowski_difference(py::cast<const ContSet &>(S), type);
                return I.minkowski_difference(vector_of(S, "the point"));
            },
            py::arg("S"), py::arg("type") = py::none(),
            "The box of x with x + S in the box; empty if there is none")
        .def(
            "unite",
            [](const Interval &I, const py::object &other) {
                if (py::isinstance<Interval>(other)) return I.unite(py::cast<Interval>(other));
                // union with a box or a point
                return I.unite(vector_of(other, "the point"));
            },
            py::arg("other"), "The smallest box around the box and a box or a point")
        .def_static("origin", &Interval::origin, py::arg("n"), "The box {0} of n dimensions")
        .def(
            "quad_map",
            [](const Interval &I, const py::list &Q) { return I.quad_map(matrix_list_of(Q)); },
            py::arg("Q"), "The box around (x' Q[i] x) for x in the box; Q[i] are (n, n) matrices")
        .def(
            "quad_map",
            [](const Interval &I, const Interval &I2, const py::list &Q) {
                return I.quad_map(I2, matrix_list_of(Q));
            },
            py::arg("I2"), py::arg("Q"), "The box around (x1' Q[i] x2) for x1 in the box, x2 in I2")
        .def("log", &Interval::log, "The image under the natural logarithm")
        .def("sin", &Interval::sin, "The image under the sine (CORA: cos(I - pi/2))")
        .def("sinh", &Interval::sinh, "The image under the hyperbolic sine")
        .def("sign", &Interval::sign, "The hull of the sign of the bounds")
        // signs and rounding
        .def(
            "round",
            [](const Interval &I, int64_t digits, const std::string &type) {
                return I.round(digits, type);
            },
            py::arg("digits") = 0, py::arg("type") = "decimals",
            "The bounds rounded to digits decimals, halves away from zero")
        .def(
            "radius", [](const Interval &I) { return scalar(I.radius()); },
            "The 2-norm of the radius over all entries")
        .def(
            "power",
            [](const Interval &I, const py::object &e) {
                if (py::isinstance<Interval>(e)) return I.power(py::cast<Interval>(e));
                if (is_number(e)) return I.power(py::cast<double>(e));
                return I.power(tensor_of(e, "the exponent"));
            },
            py::arg("exponent"),
            "The elementwise power: exponent a number, an array of the shape of the bounds or a "
            "box")
        // powers of a box
        .def_static(
            "power_of",
            [](const py::object &base, const Interval &exponent) {
                return Interval::power(operand(exponent, base, "the base"), exponent);
            },
            py::arg("base"), py::arg("exponent"), "Numbers to the power of a box")
        .def(
            "rdivide",
            [](const Interval &I, const py::object &other) {
                if (py::isinstance<Interval>(other)) return I.rdivide(py::cast<Interval>(other));
                return I.rdivide(operand(I, other, "the denominator"));
            },
            py::arg("denominator"), "The entrywise quotient with a box or numbers (CORA's ./)")
        .def_static(
            "rdivide_of",
            [](const py::object &numerator, const Interval &denominator) {
                return Interval::rdivide(operand(denominator, numerator, "the numerator"),
                                         denominator);
            },
            py::arg("numerator"), py::arg("denominator"), "Numbers over a box, entrywise")
        // products and partitions
        .def("prod", &Interval::prod, py::arg("dim") = py::none(),
             "The product of the rows (dim=1) or the columns (dim=2) of an interval matrix")
        .def("partition", &Interval::partition, py::arg("splits"),
             "The splits^n boxes of an equal grid cut of the box")
        .def("project_high_dim", &Interval::project_high_dim, py::arg("N"), py::arg("proj"),
             "The box in N dimensions at the coordinates proj (0-based), [0, 0] elsewhere")
        .def("reduce", &Interval::reduce, py::arg("order") = 1,
             "The box itself: an interval has no generators to reduce")
        .def("representsa", &Interval::representsa, py::arg("type"), py::arg("tol") = py::none(),
             "Whether the box is also a set of the CORA type, e.g. 'point', 'origin', 'zonotope'")
        .def(
            "contains",
            [](const Interval &I, const py::object &p, Tol tol) {
                return I.contains(vector_of(p, "the point"), tol);
            },
            py::arg("p"), py::arg("tol") = py::none(),
            "Whether p is in the box enlarged by tol per coordinate");
}

/// The product of an interval matrix with what Python offers: another set, a zonotope, the empty
/// set, a matrix or a number (on the right); NotImplemented for anything else.
py::object interval_matrix_product(const IntervalMatrix &A, const py::object &o) {
    if (py::isinstance<IntervalMatrix>(o)) return py::cast(A.matmul(py::cast<IntervalMatrix>(o)));
    if (py::isinstance<Zonotope>(o)) return py::cast(A.matmul(py::cast<Zonotope>(o)));
    if (py::isinstance<EmptySet>(o)) return py::cast(A.matmul(py::cast<EmptySet>(o)));
    if (is_number(o)) return py::cast(A.matmul(py::cast<double>(o)));
    if (const std::optional<Tensor> M = as_tensor(o)) return py::cast(A.matmul(*M));
    return not_implemented();
}

/// The sum of an interval matrix with what Python offers: another interval matrix, a matrix or a
/// number; NotImplemented for anything else.
py::object interval_matrix_sum(const IntervalMatrix &A, const py::object &o) {
    if (py::isinstance<IntervalMatrix>(o)) return py::cast(A.plus(py::cast<IntervalMatrix>(o)));
    if (is_number(o)) return py::cast(A.plus(py::cast<double>(o)));
    if (const std::optional<Tensor> M = as_tensor(o)) return py::cast(A.plus(*M));
    return not_implemented();
}

/// The series of expm as the object Python reads: the enclosure, the powers, their first order and
/// the remainder.
void bind_expm_results(py::module_ &m) {
    py::class_<ExpmResult>(m, "ExpmResult", "The result of IntervalMatrix.expm")
        .def_readonly("enclosure", &ExpmResult::enclosure, "The enclosure of the series")
        .def_readonly("powers", &ExpmResult::powers,
                      "The powers of A, powers[k] of order first_order + k")
        .def_readonly("first_order", &ExpmResult::first_order, "The order of powers[0]")
        .def_readonly("remainder", &ExpmResult::remainder, "The remainder box");
    py::class_<ExpmMixedResult>(m, "ExpmMixedResult", "The result of IntervalMatrix.expm_mixed")
        .def_readonly("enclosure", &ExpmMixedResult::enclosure,
                      "The terms to the intermediate order")
        .def_readonly("enclosure_rest", &ExpmMixedResult::enclosure_rest, "The rest of the series")
        .def_readonly("powers", &ExpmMixedResult::powers,
                      "The powers of orders 1 to intermediate_order")
        .def_readonly("powers2", &ExpmMixedResult::powers2, "The powers of the second part")
        .def_readonly("remainder", &ExpmMixedResult::remainder, "The remainder box");
}

void bind_interval_matrix(py::module_ &m) {
    bind_expm_results(m);
    // the interval matrix from its center and delta, or from an Interval
    py::class_<IntervalMatrix>(
        m, "IntervalMatrix",
        "An interval matrix {M | inf <= M <= sup}: C +- |D|, a point C, or an Interval")
        .def(py::init([](const py::object &center, const py::object &delta) {
                 if (py::isinstance<Interval>(center))
                     return IntervalMatrix(py::cast<Interval>(center));
                 if (delta.is_none()) return IntervalMatrix(tensor_of(center, "center"));
                 return IntervalMatrix(tensor_of(center, "center"), tensor_of(delta, "delta"));
             }),
             py::arg("center"), py::arg("delta") = py::none(),
             "The matrices C +- |D|; C alone is a point matrix; an Interval of matrix bounds is "
             "kept")
        .def_property_readonly("interval", &IntervalMatrix::interval, "The bounds, an Interval")
        .def_property_readonly("rows", &IntervalMatrix::rows, "The number of rows")
        .def_property_readonly("cols", &IntervalMatrix::cols, "The number of columns")
        .def_property_readonly(
            "center", [](const IntervalMatrix &A) { return matrix(A.center()); },
            "The center matrix (inf + sup) / 2")
        // the bounds of the interval matrix
        .def_property_readonly(
            "infimum", [](const IntervalMatrix &A) { return matrix(A.infimum()); },
            "The lower bound matrix")
        .def_property_readonly(
            "supremum", [](const IntervalMatrix &A) { return matrix(A.supremum()); },
            "The upper bound matrix")
        .def_property_readonly(
            "rad", [](const IntervalMatrix &A) { return matrix(A.rad()); },
            "The radius matrix (sup - inf) / 2")
        .def(
            "abs", [](const IntervalMatrix &A) { return matrix(A.abs()); },
            "The entrywise largest absolute value max(|inf|, |sup|)")
        .def("representsa", &IntervalMatrix::representsa, py::arg("type"),
             py::arg("tol") = py::none(),
             "Whether the entries, stacked into a vector, are a set of the CORA type, e.g. 'point'")
        .def(
            "volume", [](const IntervalMatrix &A) { return scalar(A.volume()); },
            "The product of the entry widths (0 if one is 0)")
        .def("mat_zonotope", &IntervalMatrix::mat_zonotope,
             "The same set as a matrix zonotope: one generator per entry")
        // the vertices as matrices
        .def(
            "vertices",
            [](const IntervalMatrix &A) {
                py::list matrices;
                for (const Tensor &V : A.vertices()) matrices.append(matrix(V));
                return matrices;
            },
            "The distinct vertex matrices, one (K, rows, cols) array per batch member")
        .def(
            "dim",
            [](const IntervalMatrix &A, std::optional<int> rc) -> py::object {
                if (rc) return py::cast(A.dim(*rc));
                return py::make_tuple(A.rows(), A.cols());
            },
            py::arg("rc") = py::none(),
            "(rows, cols); with rc = 1 the rows, with rc = 2 the columns")
        .def_static(
            "generate_random",
            [](cora::Rng &rng, const std::optional<std::array<int64_t, 2>> &dimension,
               const py::object &center, const py::object &max_radius) {
                // a random interval matrix
                std::optional<Tensor> c, r;
                if (!center.is_none()) c = tensor_of(center, "center");
                if (!max_radius.is_none())
                    r = is_number(max_radius)
                            ? Tensor::from_data({py::cast<double>(max_radius)}, {1, 1})
                            : tensor_of(max_radius, "max_radius");
                return IntervalMatrix::generate_random(rng, dimension, c, r);
            },
            py::arg("rng"), py::arg("dimension") = py::none(), py::arg("center") = py::none(),
            py::arg("max_radius") = py::none(),
            "A random interval matrix: the dimension (rows, cols), a center and the radii")
        .def("expm_taylor_horner_scaling", &IntervalMatrix::expm_taylor_horner_scaling,
             py::arg("max_order"), py::arg("style"), py::arg("scaling_exponent") = 1,
             "The exponential by the Taylor series (style 0), Horner (1) or scaling and squaring "
             "(2); None "
             "where the series does not converge")
        .def("expm_different_algorithms", &IntervalMatrix::expm_different_algorithms,
             py::arg("max_order"), py::arg("style"), py::arg("scaling_exponent") = 1,
             "As expm_taylor_horner_scaling, and style 4: expm at the step 1")
        // independent and mixed exponentials
        .def("expm_independent_mixed", &IntervalMatrix::expm_independent_mixed,
             py::arg("intermediate_order"), py::arg("max_order"),
             "The independent-power series in two parts, up to and beyond intermediate_order")
        .def(
            "expm_norm_infinity",
            [](const IntervalMatrix &A, double t) { return scalar(A.expm_norm_infinity(t)); },
            py::arg("t"), "CORA's closed-form bound of the exponential's remainder norm")
        .def(
            "expm_absolute_bound",
            [](const IntervalMatrix &A, double t) { return matrix(A.expm_absolute_bound(t)); },
            py::arg("t"), "CORA's matrix bound of the absolute value of the exponential")
        .def("plus", &interval_matrix_sum, py::arg("other"),
             "The sum with an interval matrix, a matrix or a number")
        .def("__add__", &interval_matrix_sum, py::is_operator())
        .def("__radd__", &interval_matrix_sum, py::is_operator())
        .def(
            "powers",
            [](const IntervalMatrix &A, int64_t max_order, std::optional<int64_t> initial_order,
               const std::optional<IntervalMatrix> &initial_power) {
                if (initial_order || initial_power) {
                    // powers with an initial order
                    if (!initial_order || !initial_power)
                        CORA_FAIL(Error::wrong_value,
                                  "powers: give initial_order and initial_power together");
                    return A.powers(max_order, *initial_order, *initial_power);
                }
                return A.powers(max_order);
            },
            py::arg("max_order"), py::arg("initial_order") = py::none(),
            py::arg("initial_power") = py::none(),
            "The powers from order initial_order (default 1) to max_order, each times A from the "
            "right")
        .def(
            "contains",
            [](const IntervalMatrix &A, const py::object &M, const std::string &type, Tol tol) {
                return A.contains(tensor_of(M, "the matrix"), type, tol);
            },
            py::arg("M"), py::arg("type") = "exact", py::arg("tol") = py::none(),
            "Whether the matrix M is in the set, entry by entry up to tol")
        .def("exact_square", &IntervalMatrix::exact_square, "The entrywise range of M @ M")
        // squares and remainders
        .def("dependent_terms", &IntervalMatrix::dependent_terms, py::arg("r"),
             "(r^2 A^2, I + A r + A^2 r^2 / 2): the exact square and the second-order terms")
        .def("exponential_remainder", &IntervalMatrix::exponential_remainder, py::arg("max_order"),
             "The remainder box of the Taylor series of the matrix exponential")
        .def(
            "expm",
            [](const IntervalMatrix &A, int64_t max_order, std::optional<double> r,
               std::optional<int64_t> initial_order,
               const std::optional<IntervalMatrix> &initial_power) {
                if (initial_order || initial_power) {
                    if (!initial_order || !initial_power)
                        CORA_FAIL(Error::wrong_value,
                                  "expm: give initial_order and initial_power together");
                    return A.expm(r.value_or(1.0), max_order, *initial_order, *initial_power);
                }
                // the exponential with optional start
                return r ? A.expm(*r, max_order) : A.expm(max_order);
            },
            py::arg("max_order") = 10, py::arg("r") = py::none(),
            py::arg("initial_order") = py::none(), py::arg("initial_power") = py::none(),
            "The enclosure of exp(M) over the set: the series to max_order with the dependent "
            "terms")
        .def(
            "expm_independent",
            [](const IntervalMatrix &A, int64_t max_order, std::optional<int64_t> initial_order,
               const std::optional<IntervalMatrix> &initial_power) {
                if (initial_order || initial_power) {
                    if (!initial_order || !initial_power)
                        CORA_FAIL(
                            Error::wrong_value,
                            "expm_independent: give initial_order and initial_power together");
                    return A.expm_independent(max_order, *initial_order, *initial_power);
                }
                // independent exponential series
                return A.expm_independent(max_order);
            },
            py::arg("max_order"), py::arg("initial_order") = py::none(),
            py::arg("initial_power") = py::none(),
            "The enclosure of exp(M) with every power evaluated independently")
        .def("expm_mixed", &IntervalMatrix::expm_mixed, py::arg("r"), py::arg("intermediate_order"),
             py::arg("max_order"), "The series in two parts, up to and beyond intermediate_order")
        .def(
            "infinity_norm", [](const IntervalMatrix &A) { return scalar(A.infinity_norm()); },
            "The largest absolute row sum of abs(A)")
        .def(
            "norm",
            [](const IntervalMatrix &A, const py::object &type) {
                return scalar(A.norm(norm_of(type)));
            },
            py::arg("type") = 2,
            "The norm of |center| + rad: 1, inf or 'fro' (2 is refused, as CORA)")
        .def("matrix_power", &IntervalMatrix::matrix_power, py::arg("exponent"),
             "The power of a square set: the repeated interval matrix product")
        .def("__pow__", &IntervalMatrix::matrix_power, py::is_operator())
        // products of interval matrices
        .def("matmul", &interval_matrix_product, py::arg("other"),
             "The product with an interval matrix, a zonotope, the empty set, a matrix or a number")
        .def("__matmul__", &interval_matrix_product, py::is_operator())
        .def("__mul__", &interval_matrix_product, py::is_operator())
        .def(
            "__rmatmul__",
            [](const IntervalMatrix &A, const py::object &o) -> py::object {
                if (const std::optional<Tensor> M = as_tensor(o))
                    return py::cast(IntervalMatrix::matmul(*M, A));
                return not_implemented();
            },
            py::is_operator())
        .def(
            "__rmul__",
            [](const IntervalMatrix &A, const py::object &o) -> py::object {
                if (is_number(o)) return py::cast(A.matmul(py::cast<double>(o)));
                if (const std::optional<Tensor> M = as_tensor(o))
                    return py::cast(IntervalMatrix::matmul(*M, A));
                // left products
                return not_implemented();
            },
            py::is_operator())
        .def("__repr__", &IntervalMatrix::display);
    m.attr("IntervalMatrix").attr("__array_ufunc__") = py::none(); // numpy defers to the operators
}

/// The product of a matrix zonotope with what Python offers on the right: another one, a zonotope,
/// a number or a matrix; NotImplemented for anything else.
py::object mat_zonotope_product(const MatZonotope &A, const py::object &o) {
    if (py::isinstance<MatZonotope>(o)) return py::cast(A.matmul(py::cast<MatZonotope>(o)));
    if (py::isinstance<Zonotope>(o)) return py::cast(A.matmul(py::cast<Zonotope>(o)));
    if (is_number(o)) return py::cast(A.matmul(py::cast<double>(o)));
    if (const std::optional<Tensor> M = as_tensor(o)) return py::cast(A.matmul(*M));
    return not_implemented();
}

void bind_mat_zonotope(py::module_ &m) {
    py::class_<MatZonotope::ExpmMixedResult>(m, "MatZonotopeExpmMixedResult",
                                             "The result of MatZonotope.expm_mixed")
        .def_readonly("enclosure", &MatZonotope::ExpmMixedResult::enclosure,
                      "The terms to the intermediate order, a MatZonotope")
        .def_readonly("enclosure_rest", &MatZonotope::ExpmMixedResult::enclosure_rest,
                      "The rest of the series, an IntervalMatrix")
        .def_readonly("powers", &MatZonotope::ExpmMixedResult::powers,
                      "The powers of orders 1 to intermediate_order")
        .def_readonly("powers2", &MatZonotope::ExpmMixedResult::powers2,
                      "The powers of the second part")
        .def_readonly("remainder", &MatZonotope::ExpmMixedResult::remainder, "The remainder box");
    // the result of the one-parameter exponential
    py::class_<MatZonotope::ExpmOneParamResult>(m, "MatZonotopeExpmOneParamResult",
                                                "The result of MatZonotope.expm_one_param")
        .def_readonly("enclosure", &MatZonotope::ExpmOneParamResult::enclosure,
                      "The matrix zonotope of the Taylor terms")
        .def_readonly("remainder", &MatZonotope::ExpmOneParamResult::remainder,
                      "The remainder box, to be added")
        .def_readonly("powers", &MatZonotope::ExpmOneParamResult::powers,
                      "The powers of r M, powers[k] of order k + 1")
        .def_readonly("constant_input", &MatZonotope::ExpmOneParamResult::constant_input,
                      "With an input, the zonotope of its terms, else None");
    py::class_<MatZonotope>(m, "MatZonotope",
                            "A matrix zonotope {C + sum b_i G_i | |b|_inf <= 1}: a set of matrices")
        .def(py::init([](const py::object &center, const py::object &generators) {
                 if (center.is_none()) return MatZonotope();
                 if (generators.is_none()) return MatZonotope(tensor_of(center, "center"));
                 // the center alone, or the center with its generators
                 return MatZonotope(tensor_of(center, "center"),
                                    tensor_of(generators, "generators"));
             }),
             py::arg("center") = py::none(), py::arg("generators") = py::none(),
             "The center C (rows, cols) and the generators G (h, rows, cols); C alone has no "
             "generators, nothing is the empty 0 x 0 set")
        .def_property_readonly(
            "center", [](const MatZonotope &A) { return matrix(A.center()); }, "The center matrix")
        .def_property_readonly(
            "generators", [](const MatZonotope &A) { return matrix(A.generators()); },
            "The generators (h, rows, cols), one matrix each")
        .def_property_readonly("rows", &MatZonotope::rows, "The number of rows")
        .def_property_readonly("cols", &MatZonotope::cols, "The number of columns")
        .def_property_readonly("num_generators", &MatZonotope::num_generators,
                               "The number of generator matrices h")
        .def("zonotope", &MatZonotope::zonotope,
             "The zonotope of the stacked matrices (column by column)")
        .def("interval_matrix", &MatZonotope::interval_matrix,
             "The box around the set, C +- sum |G_i|")
        // containment of a matrix
        .def(
            "contains",
            [](const MatZonotope &A, const py::object &M, const std::string &type, Tol tol) {
                return A.contains(tensor_of(M, "the matrix"), type, tol);
            },
            py::arg("M"), py::arg("type") = "exact", py::arg("tol") = py::none(),
            "Whether the matrix M (rows, cols), or the matrices M (p, rows, cols), are in the set")
        .def("representsa", &MatZonotope::representsa, py::arg("type"), py::arg("tol") = py::none(),
             "Whether the set is also a set of the CORA type, e.g. 'point' or 'origin'")
        .def("transpose", &MatZonotope::transpose, "The transposed matrices")
        .def(
            "powers",
            [](const MatZonotope &A, int64_t max_order, std::optional<int64_t> initial_order,
               const std::optional<MatZonotope> &initial_power) {
                if (initial_order || initial_power) {
                    if (!initial_order || !initial_power)
                        CORA_FAIL(Error::wrong_value,
                                  "powers: give initial_order and initial_power together");
                    return A.powers(max_order, *initial_order, *initial_power);
                }
                // powers with an initial order
                return A.powers(max_order);
            },
            py::arg("max_order"), py::arg("initial_order") = py::none(),
            py::arg("initial_power") = py::none(),
            "The powers of orders initial_order (default 1) to max_order, each the previous times "
            "A")
        .def(
            "norm",
            [](const MatZonotope &A, const py::object &type) {
                return scalar(A.norm(norm_of(type)));
            },
            py::arg("type") = 2,
            "An upper bound of the norm of every matrix: 1, inf or 'fro' of the box (2 is refused)")
        .def("dependent_terms", &MatZonotope::dependent_terms, py::arg("r"),
             "(r^2 A^2, I + A r + A^2 r^2 / 2) as matrix zonotopes")
        .def("expm", &MatZonotope::expm, py::arg("max_order") = 10,
             "The enclosure of exp(M) over the set (expm_independent)")
        .def("expm_independent", &MatZonotope::expm_independent, py::arg("max_order"),
             "The enclosure of exp(M) with every power evaluated independently")
        // exponentials of a matrix zonotope
        .def("expm_mixed", &MatZonotope::expm_mixed, py::arg("r"), py::arg("intermediate_order"),
             py::arg("max_order"), "The series in two parts, up to and beyond intermediate_order")
        // the exponential with independent powers, the corners, the volume and the reduction
        .def("expm_ind_mixed", &MatZonotope::expm_ind_mixed, py::arg("intermediate_order"),
             py::arg("max_order"),
             "The series in two parts with independent powers, up to and beyond intermediate_order")
        // the corner matrices of a matrix zonotope
        .def(
            "vertices",
            [](const MatZonotope &A) {
                py::list matrices;
                for (const Tensor &V : A.vertices()) matrices.append(matrix(V));
                return matrices;
            },
            "The corner matrices, one (K, rows, cols) array per batch member")
        // the volume and the reduction go through the stacked zonotope
        .def(
            "volume",
            [](const MatZonotope &A, const std::string &method, int order) {
                return scalar(A.volume(method, order));
            },
            py::arg("method") = "exact", py::arg("order") = 5,
            "The volume of the zonotope of the stacked matrices")
        .def("reduce", &MatZonotope::reduce, py::arg("order") = 1, py::arg("method") = "girard",
             "A matrix zonotope with fewer generators that contains this one (Girard)")
        .def(
            "expm_one_param",
            [](const MatZonotope &A, double r, int64_t max_order, const py::object &u_trans) {
                if (u_trans.is_none()) return A.expm_one_param(r, max_order);
                if (py::isinstance<Zonotope>(u_trans))
                    return A.expm_one_param(r, max_order, py::cast<Zonotope>(u_trans));
                const Tensor u = tensor_of(u_trans, "u_trans"); // a column is the zonotope (u, 0)
                return A.expm_one_param(r, max_order, Zonotope(u, u.zeros_like()));
            },
            py::arg("r"), py::arg("max_order"), py::arg("u_trans") = py::none(),
            "The Taylor series of exp(M r) for the matrices C + b G of a set with one generator, "
            "with the terms of an input u_trans (a zonotope with one generator or a vector)")
        .def("__matmul__", &mat_zonotope_product, py::is_operator())
        .def("__mul__", &mat_zonotope_product, py::is_operator())
        .def(
            "__rmatmul__",
            [](const MatZonotope &A, const py::object &o) -> py::object {
                if (const std::optional<Tensor> M = as_tensor(o))
                    return py::cast(MatZonotope::matmul(*M, A));
                return not_implemented();
            },
            py::is_operator())
        .def(
            "__rmul__",
            [](const MatZonotope &A, const py::object &o) -> py::object {
                if (is_number(o)) return py::cast(A.matmul(py::cast<double>(o)));
                // a number or a matrix on the left of a matrix zonotope
                if (const std::optional<Tensor> M = as_tensor(o))
                    return py::cast(MatZonotope::matmul(*M, A));
                return not_implemented();
            },
            py::is_operator())
        .def(
            "__pow__", [](const MatZonotope &A, int64_t k) { return A.matrix_power(k); },
            py::is_operator())
        .def(
            "__add__",
            [](const MatZonotope &A, const py::object &o) -> py::object {
                if (py::isinstance<MatZonotope>(o))
                    return py::cast(A.plus(py::cast<MatZonotope>(o)));
                if (const std::optional<Tensor> M = as_tensor(o)) return py::cast(A.plus(*M));
                return not_implemented();
            },
            py::is_operator())
        .def(
            "__radd__",
            [](const MatZonotope &A, const py::object &o) -> py::object {
                // a matrix on the left is added
                if (const std::optional<Tensor> M = as_tensor(o)) return py::cast(A.plus(*M));
                return not_implemented();
            },
            py::is_operator())
        .def("__repr__", [](const MatZonotope &A) {
            return "MatZonotope(" + std::to_string(A.rows()) + " x " + std::to_string(A.cols()) +
                   ", " + std::to_string(A.num_generators()) + " generators)";
        });
    m.attr("MatZonotope").attr("__array_ufunc__") = py::none(); // numpy defers to the operators
}

/// The set without points: what Python adds to the inherited ContSet methods.
void bind_empty_set(py::module_ &m) {
    py::class_<EmptySet, ContSet>(m, "EmptySet", "The empty set of n dimensions.")
        .def(py::init([](const std::optional<int64_t> &n) {
                 if (!n)
                     CORA_FAIL(Error::no_input_in_set_constructor,
                               "EmptySet: give the dimension, e.g. EmptySet(2)");
                 return EmptySet(*n);
             }),
             py::arg("n") = py::none(), "The empty set of n >= 0 dimensions")
        .def_static("empty", &EmptySet::empty, py::arg("n") = 0, "The empty set of n dimensions")
        .def("is_bounded", &EmptySet::is_bounded, "Whether the set is bounded: always")
        .def("intersect", &EmptySet::intersect, py::arg("S"), "The intersection with S: empty")
        .def(
            "plus",
            [](const EmptySet &E, const py::object &S) {
                if (py::isinstance<ContSet>(S)) return E.plus(py::cast<const ContSet &>(S));
                // the sum with a set or with a vector is empty
                return E.plus(vector_of(S, "the vector"));
            },
            py::arg("S"), "The sum with a set or a vector: empty")
        .def(
            "convex_hull",
            [](const EmptySet &E, const py::object &S) {
                return object_of(E.convex_hull(set_of(S)));
            },
            py::arg("S"), "The convex hull with S: S itself")
        .def(
            "contains",
            [](const EmptySet &E, const py::object &S, Tol tol) {
                if (py::isinstance<ContSet>(S))
                    return E.contains(py::cast<const ContSet &>(S), tol);
                return E.contains(tensor_of(S, "the points"), tol);
            },
            py::arg("S"), py::arg("tol") = py::none(),
            "Whether S (a set, or points (n, N)) is contained: only if it is empty")
        .def("is_intersecting", &EmptySet::is_intersecting, py::arg("S"),
             py::arg("tol") = py::none(), "Whether S meets the set: never")
        // the set is empty and the sum equals the other
        .def("isequal", &EmptySet::isequal, py::arg("S"), py::arg("tol") = py::none(),
             "Whether S is the empty set of the same dimension")
        .def(
            "linear_map",
            [](const EmptySet &E, const py::object &M) {
                if (is_number(M)) return E.linear_map(py::cast<double>(M));
                if (py::isinstance<Interval>(M)) return E.linear_map(py::cast<Interval>(M));
                return E.linear_map(tensor_of(M, "the matrix"));
            },
            py::arg("M"), "The image under a matrix, an interval matrix or a number: empty")
        .def("project_high_dim", &EmptySet::project_high_dim, py::arg("N"), py::arg("proj"),
             "The empty set in N dimensions, with the checks of lift")
        .def_static("generate_random", &EmptySet::generate_random, py::arg("rng"),
                    py::arg("dimension") = py::none(),
                    "The empty set of the dimension, or of a random one in 1..10")
        .def("radius", &EmptySet::radius, "The radius of the smallest ball around it: 0")
        .def("volume", &EmptySet::volume, "The volume: 0")
        .def("complement", &EmptySet::complement, "The complement: the whole space")
        // set predicates of the empty set
        .def(
            "representsa", &EmptySet::representsa, py::arg("type"), py::arg("tol") = py::none(),
            "Whether the set is also a set of CORA's type: true for all that have an empty member");
}

/// The matrices of a cubic map T[i][k] from Python: a list of rows, each a list of arrays.
std::vector<std::vector<Tensor>> matrices_of(const py::list &T) {
    std::vector<std::vector<Tensor>> out;
    for (const py::handle &row : T) {
        out.emplace_back();
        for (const py::handle &M : py::reinterpret_borrow<py::sequence>(row))
            out.back().push_back(tensor_of(M, "a matrix of T"));
    }
    return out;
}

using Indices = std::vector<std::vector<int64_t>>;

void bind_zonotope(py::module_ &m) {
    py::class_<Zonotope, ContSet>(m, "Zonotope", "The zonotope {c + G b : |b|_inf <= 1}.")
        .def(py::init([](const Array &c, const Array &G) {
                 // the zonotope from center and generators
                 return Zonotope(vector_of(c, "the center"), tensor_of(G, "the generators"));
             }),
             py::arg("center"), py::arg("generators"), "center (..., n) and generators (..., n, m)")
        // the list filter and the conversions of a zonotope
        .def_static("filter_out", &Zonotope::filter_out, py::arg("zonotopes"),
                    "The indices (0-based) of the zonotopes that no other zonotope contains")
        // the simplex around it, and the matrix zonotope of its columns
        .def("simplex", &Zonotope::simplex, "A simplex (as a Polytope) that contains the zonotope")
        .def("mat_zonotope", &Zonotope::mat_zonotope,
             "The same set as a matrix zonotope of n x 1 matrices")
        .def_static("stack", &Zonotope::stack, py::arg("zonotopes"),
                    "A batch of zonotopes as one; fewer generators are padded with zeros")
        .def_static(
            "generate_random",
            [](cora::Rng &rng, const std::optional<int64_t> &dimension, const py::object &center,
               const std::optional<int64_t> &nr_generators, const std::string &distribution) {
                return Zonotope::generate_random(rng, dimension, optional_vector(center, "center"),
                                                 nr_generators, distribution);
            },
            py::arg("rng"), py::arg("dimension") = py::none(), py::arg("center") = py::none(),
            py::arg("nr_generators") = py::none(), py::arg("distribution") = "uniform",
            "A random zonotope: the dimension (default that of center, else random in 1..10), a "
            "center, the number of generators (default 2 * dimension) and the distribution of "
            "their lengths ('uniform', 'exp' or 'gamma')")
        .def_property_readonly(
            "center", [](const Zonotope &Z) { return vector(Z.center()); }, "The center c (..., n)")
        // the center and the generators of the zonotope
        .def_property_readonly(
            "generators", [](const Zonotope &Z) { return matrix(Z.generators()); },
            "The generator matrix G (..., n, m)")
        .def(
            "rand_point",
            [](const Zonotope &Z, int64_t N, cora::Rng &rng, const std::string &type) {
                return matrix(Z.rand_point(N, rng, type));
            },
            py::arg("N"), py::arg("rng"), py::arg("type") = "standard",
            "N random points as columns (..., n, N); type 'standard' or 'extreme'")
        .def(
            "linear_map",
            [](const Zonotope &Z, const py::object &M) {
                if (py::isinstance<Interval>(M)) return Z.linear_map(py::cast<Interval>(M));
                return Z.linear_map(tensor_of(M, "the matrix"));
            },
            py::arg("M"), "M Z for a matrix (..., k, n), or an interval matrix")
        .def("plus", py::overload_cast<const Zonotope &>(&Zonotope::plus, py::const_),
             py::arg("Z2"), "The Minkowski sum with a zonotope or a box")
        // sums with other sets
        .def("plus", py::overload_cast<const Interval &>(&Zonotope::plus, py::const_), py::arg("I"),
             "The Minkowski sum with a zonotope or a box")
        .def("linear_combination", &Zonotope::linear_combination, py::arg("Z2"),
             "The zonotope of the convex combinations of this one and Z2")
        .def("reduce", &Zonotope::reduce, py::arg("order") = 1,
             "At most order * n generators; the smallest are enclosed by a box (Girard)")
        // the reduction by the share of the box
        .def("reduce_adaptive", &Zonotope::reduce_adaptive, py::arg("diag_percent"),
             "The generators of least Girard metric within diag_percent of the box's norm are "
             "enclosed by a box (CORA's reduce(Z, 'adaptive', diag_percent))")
        .def("reduce_pca", &Zonotope::reduce_pca,
             "The box of the generators in their principal axes (CORA's reduce(Z, 'pca', 1))")
        .def("abs", &Zonotope::abs, "|c| and |G| entrywise (not the image of the set)")
        .def("box", &Zonotope::box, "The box around the zonotope, as a zonotope")
        .def("compact", &Zonotope::compact, py::arg("method") = py::none(),
             py::arg("tol") = py::none(),
             "Without zero generators ('zeros'); 'all' also merges aligned ones; 'aligned': as is")
        .def(
            "cart_prod",
            [](const Zonotope &Z, const py::object &S) {
                if (py::isinstance<Zonotope>(S)) return Z.cart_prod(py::cast<Zonotope>(S));
                if (py::isinstance<Interval>(S)) return Z.cart_prod(py::cast<Interval>(S));
                return Z.cart_prod(vector_of(S, "the point"));
            },
            py::arg("S"), "The Cartesian product with a zonotope, a box or a point")
        // enclose with a set
        .def(
            "enclose",
            [](const Zonotope &Z, const ContSet &S2) {
                return object_of(static_cast<const ContSet &>(Z).enclose(S2));
            },
            py::arg("S2"), "A zonotope that encloses both")
        .def(
            "enclose",
            [](const Zonotope &Z, const py::object &M, const Zonotope &Zplus) {
                return Z.enclose(tensor_of(M, "the matrix"), Zplus);
            },
            py::arg("M"), py::arg("Zplus"), "A zonotope that encloses Z and M Z + Zplus")
        .def(
            "convex_hull",
            [](const Zonotope &Z, const py::object &S) {
                if (py::isinstance<Zonotope>(S)) return Z.convex_hull(py::cast<Zonotope>(S));
                if (py::isinstance<Interval>(S)) return Z.convex_hull(py::cast<Interval>(S));
                return Z.convex_hull(vector_of(S, "the point"));
            },
            py::arg("S"), "A zonotope that encloses the convex hull with a zonotope, box or point")
        // intersect with a box or a zonotope
        .def(
            "intersect",
            [](const Zonotope &Z, const py::object &S, const std::string &method) {
                if (py::isinstance<Interval>(S)) return Z.intersect(py::cast<Interval>(S), method);
                return Z.intersect(py::cast<Zonotope>(S), method);
            },
            py::arg("S"), py::arg("method") = "conZonotope",
            "A zonotope that encloses the intersection: boxes exactly, else method 'averaging'")
        .def(
            "constraints_satisfied",
            [](const Zonotope &Z, const py::object &C, const py::object &d) {
                return Z.constraints_satisfied(tensor_of(C, "C"), vector_of(d, "d"));
            },
            py::arg("C"), py::arg("d"), "True if C x < d is certain for the whole zonotope")
        .def(
            "cubic_map",
            [](const Zonotope &Z, const py::list &T, const std::optional<Indices> &ind) {
                // the cubic map
                return Z.cubic_map(matrices_of(T), ind.value_or(Indices{}));
            },
            py::arg("T"), py::arg("ind") = py::none(),
            "An enclosure of z_i = sum_k x_k (x' T[i][k] x); ind[i] lists the k of output i")
        .def(
            "cubic_map",
            [](const Zonotope &Z, const Zonotope &Z2, const Zonotope &Z3, const py::list &T,
               const std::optional<Indices> &ind) {
                return Z.cubic_map(Z2, Z3, matrices_of(T), ind.value_or(Indices{}));
            },
            py::arg("Z2"), py::arg("Z3"), py::arg("T"), py::arg("ind") = py::none(),
            "An enclosure of (x1' T[i] x2) x3 for x1 in Z, x2 in Z2, x3 in Z3")
        .def(
            "hausdorff_distance_to_box",
            [](const Zonotope &Z, const std::optional<std::string> &method) {
                return scalar(Z.hausdorff_distance_to_box(method));
            },
            py::arg("method") = py::none(),
            "A bound on the Hausdorff distance to the box: 'exact' (n <= 3), 'naive', 'wgreedy', "
            "'wopt'")
        // dominant directions
        .def(
            "dominant_directions",
            [](const Zonotope &Z, const std::optional<int> &filter_length1,
               const std::optional<int> &filter_length2) {
                return matrix(Z.dominant_directions(filter_length1, filter_length2));
            },
            py::arg("filter_length1") = py::none(), py::arg("filter_length2") = py::none(),
            "n generators (columns) that span a tight parallelotope around the zonotope")
        .def(
            "tensor_multiplication",
            [](const Zonotope &Z, const py::object &M, const Indices &perm, const Indices &comb,
               const std::vector<int64_t> &map) {
                return Z.tensor_multiplication(tensor_of(M, "M"), {perm, comb, map});
            },
            py::arg("M"), py::arg("perm"), py::arg("comb"), py::arg("map"),
            "(0, H) enclosing {sum_jkl M[i, j, k, l] x_j x_k x_l} for a zonotope centered at 0; "
            "perm (ordered generator triples), comb (its row count: the generators of H) and map")
        // tensor multiplication
        .def(
            "tensor_multiplication_zono",
            [](const Zonotope &Z, const py::object &M_center, const py::list &M_generators,
               const Indices &perm, const Indices &comb, const std::vector<int64_t> &map) {
                std::vector<Tensor> generators;
                for (const py::handle &M : M_generators)
                    generators.push_back(tensor_of(M, "a generator of M"));
                return Z.tensor_multiplication_zono(tensor_of(M_center, "M_center"), generators,
                                                    {perm, comb, map});
            },
            py::arg("M_center"), py::arg("M_generators"), py::arg("perm"), py::arg("comb"),
            py::arg("map"), "tensor_multiplication for a tensor zonotope: the sum over its parts")
        .def(
            "underapproximate",
            [](const Zonotope &Z, const py::object &S) {
                return matrix(S.is_none() ? Z.underapproximate()
                                          : Z.underapproximate(tensor_of(S, "S")));
            },
            py::arg("S") = py::none(),
            "Vertices c +- sum_j sign(s_i' g_j) g_j, columns (n, 2k) for the directions S (n, k)")
        // the volume
        .def(
            "volume",
            [](const Zonotope &Z, const std::string &method, int order) {
                return scalar(Z.volume(method, order));
            },
            py::arg("method") = "exact", py::arg("order") = 5,
            "The volume: 'exact' (sum over the n-subsets of generators) or 'alamo' (a lower bound)")
        .def(
            "zonotope_norm",
            [](const Zonotope &Z, const py::object &p) {
                return scalar(Z.zonotope_norm(vector_of(p, "p")));
            },
            py::arg("p"), "The norm of p with the zonotope centered at 0 as unit ball")
        .def(
            "zonotope_norm_minimizer",
            [](const Zonotope &Z, const py::object &p) {
                const auto [value, beta] = Z.zonotope_norm_minimizer(vector_of(p, "p"));
                return py::make_tuple(scalar(value), vector(beta));
            },
            py::arg("p"), "(norm, beta): the norm of p and the coefficients that reach it")
        // containment and norm minimizers
        .def(
            "contains",
            [](const Zonotope &Z, const py::object &S, const std::string &method, Tol tol) {
                if (py::isinstance<Zonotope>(S))
                    return Z.contains(py::cast<Zonotope>(S), method, tol);
                if (py::isinstance<Interval>(S))
                    return Z.contains(py::cast<Interval>(S), method, tol);
                return Z.contains(tensor_of(S, "the points"), method, tol);
            },
            py::arg("S"), py::arg("method") = "exact", py::arg("tol") = py::none(),
            "Whether the zonotope contains the point(s) (n,) or (n, N), a zonotope or a box")
        .def(
            "boundary_point",
            [](const Zonotope &Z, const py::object &dir, const std::optional<py::object> &start) {
                return vector(
                    start ? Z.boundary_point(vector_of(dir, "dir"), vector_of(*start, "start"))
                          : Z.boundary_point(vector_of(dir, "dir")));
            },
            py::arg("dir"), py::arg("start") = py::none(),
            "Where the ray from start (default: the center) in direction dir leaves the zonotope")
        // exact sums
        .def(
            "exact_plus",
            [](const Zonotope &Z, const Zonotope &Z2, const std::optional<int64_t> &k) {
                return Z.exact_plus(Z2, k);
            },
            py::arg("Z2"), py::arg("k") = py::none(),
            "The sum with the first min(m, m2, k) generators added pairwise (a subset of the plus)")
        .def(
            "generator_length",
            [](const Zonotope &Z) { return matrix(Z.generator_length().squeeze(-2)); },
            "The Euclidean length of every generator, (..., m)")
        .def("is_bounded", &Zonotope::is_bounded, "Always true")
        .def("is_full_dim", &Zonotope::is_full_dim, py::arg("tol") = py::none(),
             "Whether the generators span the space (singular values up to tol count as zero)")
        .def(
            "is_full_dim_subspace",
            [](const Zonotope &Z, Tol tol) {
                const auto [res, subspace] = Z.is_full_dim_subspace(tol);
                // the full-dimensional subspace
                return py::make_tuple(res, matrix(subspace));
            },
            py::arg("tol") = py::none(),
            "(res, subspace): is_full_dim and a basis (n, r) of the span")
        .def(
            "isequal",
            [](const Zonotope &Z, const py::object &S, Tol tol) {
                if (py::isinstance<Zonotope>(S)) return Z.isequal(py::cast<Zonotope>(S), tol);
                if (py::isinstance<Interval>(S)) return Z.isequal(py::cast<Interval>(S), tol);
                return Z.isequal(tensor_of(S, "the point"), tol);
            },
            py::arg("S"), py::arg("tol") = py::none(),
            "Whether S (a zonotope, a box or a point) is the same set, up to tol")
        .def(
            "is_intersecting",
            [](const Zonotope &Z, const py::object &S, const std::string &type, Tol tol) {
                if (py::isinstance<Zonotope>(S))
                    return Z.is_intersecting(py::cast<Zonotope>(S), type, tol);
                if (py::isinstance<Interval>(S))
                    return Z.is_intersecting(py::cast<Interval>(S), type, tol);
                // intersection with sets or points
                return Z.is_intersecting(tensor_of(S, "the points"), type, tol);
            },
            py::arg("S"), py::arg("type") = "exact", py::arg("tol") = py::none(),
            "Whether Z and a zonotope, a box or points intersect; 'exact' (default) or 'approx'")
        .def(
            "minnorm",
            [](const Zonotope &Z) {
                const auto [value, x] = Z.minnorm();
                return py::make_tuple(scalar(value), vector(x));
            },
            "(val, x): the distance from the center to the boundary and the nearest boundary point")
        .def(
            "intersect_strip",
            [](const Zonotope &Z, const py::object &C, const py::object &phi, const py::object &y,
               const py::object &method) {
                const Tensor Ct = tensor_of(C, "C"), pt = vector_of(phi, "phi"),
                             yt = vector_of(y, "y");
                if (py::isinstance<py::str>(method))
                    return Z.intersect_strip(Ct, pt, yt, py::cast<std::string>(method));
                if (py::isinstance<py::dict>(method)) {
                    // the options of a dict
                    const py::dict d = py::cast<py::dict>(method);
                    const auto field = [&](const char *key) {
                        if (!d.contains(key))
                            CORA_FAIL(Error::wrong_field_value,
                                      std::string("intersect_strip: the observer needs the key '") +
                                          key + "'");
                        return tensor_of(d[key], key);
                    };
                    return Z.intersect_strip(Ct, pt, yt,
                                             {field("E"), field("F"), field("A"), field("C")});
                }
                return Z.intersect_strip(Ct, pt, yt, tensor_of(method, "the gain"));
            },
            py::arg("C"), py::arg("phi"), py::arg("y"), py::arg("method") = "normGen",
            "Encloses the intersection with |C x - y| <= phi; method: a name, a gain (n, k) or "
            "a dict with the observer matrices E, F, A, C ('wang-FRad')")
        .def_static(
            "enclose_points",
            [](const py::object &points, const py::object &method) {
                // enclose points with a method
                if (py::isinstance<py::str>(method))
                    return Zonotope::enclose_points(tensor_of(points, "the points"),
                                                    py::cast<std::string>(method));
                return Zonotope::enclose_points(tensor_of(points, "the points"),
                                                tensor_of(method, "the frame U"));
            },
            py::arg("points"), py::arg("method") = "maiga",
            "A zonotope containing the points (n, N): 'maiga', 'stursberg', or the frame U (n, n)")
        .def_static(
            "enclose_many",
            [](const std::vector<Zonotope> &Zs, const py::object &direction) {
                const auto [Z, inverse] =
                    Zonotope::enclose_many(Zs, vector_of(direction, "direction"));
                return py::make_tuple(Z, matrix(inverse));
            },
            py::arg("zonotopes"), py::arg("direction"),
            "(Z, rot_inv): a box oriented along the direction around the zonotopes, and the "
            "inverse of its frame")
        // Minkowski difference
        .def(
            "mink_diff",
            [](const Zonotope &Z, const py::object &S, const std::string &method) {
                if (py::isinstance<Zonotope>(S)) return Z.mink_diff(py::cast<Zonotope>(S), method);
                if (py::isinstance<Interval>(S)) return Z.mink_diff(py::cast<Interval>(S), method);
                return Z.mink_diff(vector_of(S, "the point"), method);
            },
            py::arg("S"), py::arg("method") = "approx",
            "The Minkowski difference with a zonotope, a box or a point; methods: approx, inner, "
            "exact, outer, outer:coarse")
        .def_static("origin", &Zonotope::origin, py::arg("n"), "The zonotope {0} of n dimensions")
        .def(
            "radius", [](const Zonotope &Z) { return scalar(Z.radius()); },
            "A bound on the distance from the center: the norm of the radius of the box")
        .def("rank", &Zonotope::rank, py::arg("tol") = py::none(),
             "The rank of the generators; singular values up to tol count as zero")
        .def("representsa", &Zonotope::representsa, py::arg("type"), py::arg("tol") = py::none(),
             "Whether the zonotope is also a set of the CORA type ('origin', 'point', 'interval', "
             "'parallelotope', 'capsule', 'zonotope', ...)")
        // projection and rotation
        .def("project_high_dim", &Zonotope::project_high_dim, py::arg("N"), py::arg("dims"),
             "In N dimensions, with state i at position dims[i] and zeros elsewhere")
        .def("rotate", &Zonotope::rotate, py::arg("dims"), py::arg("angle"),
             "Turned by angle (radians) in the plane of the two states dims")
        .def(
            "quad_map",
            [](const Zonotope &Z, const py::list &Q) { return Z.quad_map(matrix_list_of(Q)); },
            py::arg("Q"), "An enclosure of (x' Q[i] x) for x in Z; Q[i] are (n, n) matrices")
        .def(
            "quad_map",
            [](const Zonotope &Z, const Zonotope &Z2, const py::list &Q) {
                return Z.quad_map(Z2, matrix_list_of(Q));
            },
            py::arg("Z2"), py::arg("Q"), "An enclosure of (x1' Q[i] x2) for x1 in Z, x2 in Z2")
        .def("split", py::overload_cast<>(&Zonotope::split, py::const_),
             "One pair of halves of the box around Z per coordinate")
        .def("split", py::overload_cast<int64_t>(&Zonotope::split, py::const_), py::arg("N"),
             "The halves of the box around Z cut at the center of coordinate N")
        .def("split", py::overload_cast<int64_t, int>(&Zonotope::split, py::const_), py::arg("N"),
             py::arg("max_order"), "The halves along generator N of Z reduced to max_order")
        // splitting
        .def(
            "split",
            [](const Zonotope &Z, const py::object &dir) { return Z.split(vector_of(dir, "dir")); },
            py::arg("dir"), "The halves cut by a plane through the center normal to dir")
        // the number overload first: an object parameter would take it as aux_dir
        .def(
            "split",
            [](const Zonotope &Z, const py::object &a, double d) {
                return Z.split(vector_of(a, "a"), d);
            },
            py::arg("a"), py::arg("d"),
            "The pieces on the two sides of the plane a'x = d, or Z alone if the plane misses it")
        .def(
            "split",
            [](const Zonotope &Z, const py::object &orig_dir, const py::object &aux_dir) {
                // split along a direction
                return Z.split(vector_of(orig_dir, "orig_dir"), vector_of(aux_dir, "aux_dir"));
            },
            py::arg("orig_dir"), py::arg("aux_dir"),
            "The halves cut normal to aux_dir without its part along orig_dir")
        .def("split_first_gen", &Zonotope::split_first_gen,
             "The halves along the generator of the largest sum |g_j' g_k|")
        .def("split_longest_gen", &Zonotope::split_longest_gen,
             "The halves along the longest generator, c + g/2 first")
        .def("reduce_under_approx", &Zonotope::reduce_under_approx, py::arg("method"),
             py::arg("order"),
             "A zonotope inside Z with at most order * n generators: 'yang', 'raghuraman', "
             "'kochdumper'")
        .def(
            "unite",
            [](const Zonotope &Z, const py::object &S, const std::string &method,
               const std::optional<int> &order) {
                if (py::isinstance<Zonotope>(S))
                    return Z.unite(py::cast<Zonotope>(S), method, order);
                if (py::isinstance<Interval>(S))
                    return Z.unite(py::cast<Interval>(S), method, order);
                // a list of zonotopes is a union
                if (py::isinstance<py::list>(S) || py::isinstance<py::tuple>(S))
                    return Z.unite(py::cast<std::vector<Zonotope>>(S), method, order);
                return Z.unite(vector_of(S, "the point"), method, order);
            },
            py::arg("S"), py::arg("method") = "linprog", py::arg("order") = py::none(),
            "A zonotope that contains the union with S (a zonotope, a box, a point or a list of "
            "zonotopes); methods linprog, iterative, parallelotope");
}

/// The operators with numbers and arrays, as CORA writes them: s * S, M * S and M @ S (the linear
/// map), S + v, S - v, -S. What an operator does not understand it hands back to Python
/// (NotImplemented), which raises the TypeError.
// ConZonotope -------------------------------------------------------------------------------------

/// A constrained zonotope from Python: a ConZonotope, or a zonotope or a box as one.
ConZonotope con_zonotope_of(const py::handle &o) {
    if (py::isinstance<ConZonotope>(o)) return py::cast<ConZonotope>(o);
    if (py::isinstance<Zonotope>(o)) return ConZonotope(py::cast<Zonotope>(o));
    if (py::isinstance<Interval>(o)) return ConZonotope(py::cast<Interval>(o));
    CORA_FAIL(Error::wrong_value, "cora: expected a ConZonotope, a Zonotope or an Interval");
}

void bind_con_zonotope(py::module_ &m) {
    py::class_<ConZonotope, ContSet>(m, "ConZonotope",
                                     "The constrained zonotope {c + G x : |x|_inf <= 1, A x = b}.")
        .def(py::init([](const py::object &c, const py::object &G, const py::object &A,
                         const py::object &b) {
                 if (A.is_none() != b.is_none())
                     CORA_FAIL(Error::wrong_input_in_constructor,
                               "ConZonotope: give the constraint matrix and the offset together");
                 if (A.is_none()) return ConZonotope(vector_of(c, "the center"), tensor_of(G, "G"));
                 // constraints are given together or not at all
                 return ConZonotope(vector_of(c, "the center"), tensor_of(G, "G"),
                                    tensor_of(A, "A"), vector_of(b, "b"));
             }),
             py::arg("c"), py::arg("G"), py::arg("A") = py::none(), py::arg("b") = py::none(),
             "center c (..., n), generators G (..., n, m) and the constraints A x = b")
        .def(py::init<const Zonotope &>(), py::arg("Z"), "A zonotope as a constrained zonotope")
        .def(py::init<const Interval &>(), py::arg("I"), "A box as a constrained zonotope")
        .def_static(
            "from_matrix",
            [](const py::object &Z, const py::object &A, const py::object &b) {
                if (A.is_none() != b.is_none())
                    CORA_FAIL(Error::wrong_input_in_constructor,
                              "ConZonotope: give the constraint matrix and the offset together");
                return ConZonotope::from_matrix(
                    tensor_of(Z, "Z"),
                    A.is_none() ? std::nullopt : std::optional<Tensor>(tensor_of(A, "A")),
                    b.is_none() ? std::nullopt : std::optional<Tensor>(vector_of(b, "b")));
            },
            py::arg("Z"), py::arg("A") = py::none(), py::arg("b") = py::none(),
            "CORA's legacy form: Z = [c G] (..., n, 1 + m)")
        // the empty set is a constrained zonotope without generators
        .def_static("empty", &ConZonotope::empty, py::arg("n") = 0,
                    "The empty set of n dimensions: no generators and the constraint 0 = 1")
        .def_property_readonly(
            "c", [](const ConZonotope &Z) { return vector(Z.c()); }, "The stored center c (..., n)")
        .def_property_readonly(
            "generators", [](const ConZonotope &Z) { return matrix(Z.generators()); },
            "The generators (..., n, m)")
        .def_property_readonly(
            "A", [](const ConZonotope &Z) { return matrix(Z.A()); },
            "The constraint matrix (..., p, m)")
        .def_property_readonly(
            "b", [](const ConZonotope &Z) { return vector(Z.b()); },
            "The constraint offset (..., p)")
        .def("is_bounded", &ConZonotope::is_bounded, "Always true")
        .def("is_full_dim", &ConZonotope::is_full_dim, py::arg("tol") = py::none(),
             "Whether the set has a nonempty interior")
        .def(
            "is_full_dim_subspace",
            [](const ConZonotope &Z, Tol tol) {
                const auto [res, subspace] = Z.is_full_dim_subspace(tol);
                // the full-dimensional subspace
                return py::make_tuple(res, matrix(subspace));
            },
            py::arg("tol") = py::none(),
            "(res, subspace): is_full_dim and a basis (n, r) of the span")
        .def(
            "contains",
            [](const ConZonotope &Z, const py::object &S, const std::string &method, Tol tol) {
                if (py::isinstance<Zonotope>(S))
                    return Z.contains(py::cast<Zonotope>(S), method, tol);
                if (py::isinstance<Interval>(S))
                    return Z.contains(py::cast<Interval>(S), method, tol);
                return Z.contains(tensor_of(S, "the points"), method, tol);
            },
            py::arg("S"), py::arg("method") = "exact", py::arg("tol") = py::none(),
            "Whether the set contains the points (n, N), a zonotope or a box; method 'exact'")
        .def_static("origin", &ConZonotope::origin, py::arg("n"), "The set {0} of n dimensions")
        // a random set with its constraints
        .def_static(
            "generate_random",
            [](cora::Rng &rng, std::optional<int64_t> dimension, const py::object &center,
               std::optional<int64_t> nr_generators, std::optional<int64_t> nr_constraints) {
                return ConZonotope::generate_random(rng, dimension,
                                                    optional_vector(center, "center"),
                                                    nr_generators, nr_constraints);
            },
            py::arg("rng"), py::arg("dimension") = py::none(), py::arg("center") = py::none(),
            py::arg("nr_generators") = py::none(), py::arg("nr_constraints") = py::none(),
            "A random set: dimension (default that of center, else 1..10), center, nr_generators "
            "(default 2 * dimension) and nr_constraints (default random)")
        // the conversions of a constrained zonotope
        .def("zonotope", &ConZonotope::zonotope, py::arg("method") = "nullSpace",
             "A zonotope that contains the set; method 'nullSpace' (default) or 'reduce'")
        // the other forms of the same set
        .def("zono_bundle", &ConZonotope::zono_bundle, "The same set as a zonotope bundle")
        .def("polytope", &ConZonotope::polytope, "The same set in halfspace form")
        // random points of either type
        .def(
            "rand_point",
            [](const ConZonotope &Z, int64_t N, cora::Rng &rng, const std::string &type) {
                return matrix(Z.rand_point(N, rng, type));
            },
            py::arg("N"), py::arg("rng"), py::arg("type") = "standard",
            "N random points as columns (n, N); type 'standard' or 'extreme'")
        .def(
            "plus",
            [](const ConZonotope &Z, const py::object &S) {
                if (py::isinstance<ContSet>(S)) return Z.plus(con_zonotope_of(S));
                // sums
                return Z.plus(vector_of(S, "the point"));
            },
            py::arg("S"),
            "The Minkowski sum with a constrained zonotope, a zonotope, a box or a point")
        .def(
            "reduce",
            [](const ConZonotope &Z, int order, const std::string &method) {
                return Z.reduce(order, method);
            },
            py::arg("order") = 1, py::arg("method") = "girard",
            "A set that encloses Z with about order * dim generators (Girard's rule on [G; A])")
        .def(
            "quad_map",
            [](const ConZonotope &Z, const py::list &Q) { return Z.quad_map(matrix_list_of(Q)); },
            py::arg("Q"), "An enclosure of (x' Q[i] x) for x in Z; Q[i] are (n, n) matrices")
        .def(
            "quad_map",
            [](const ConZonotope &Z, const ConZonotope &Z2, const py::list &Q) {
                // quadratic maps
                return Z.quad_map(Z2, matrix_list_of(Q));
            },
            py::arg("Z2"), py::arg("Q"), "An enclosure of (x1' Q[i] x2) for x1 in Z, x2 in Z2")
        .def("rescale", &ConZonotope::rescale, py::arg("method") = "exact",
             "The same set with every factor scaled to its range; method 'exact' or 'iter'")
        .def("representsa", &ConZonotope::representsa, py::arg("type"), py::arg("tol") = py::none(),
             "Whether the set is also a set of the CORA type ('origin', 'point', 'interval', "
             "'zonotope', 'emptySet', ...)")
        .def(
            "isequal",
            [](const ConZonotope &Z, const py::object &S, Tol tol) {
                if (py::isinstance<Interval>(S)) return Z.isequal(py::cast<Interval>(S), tol);
                return Z.isequal(con_zonotope_of(S), tol);
            },
            py::arg("S"), py::arg("tol") = py::none(),
            "Whether S (a constrained zonotope, a zonotope or a box) is the same set, up to tol")
        .def(
            "is_intersecting",
            [](const ConZonotope &Z, const py::object &S, const std::string &type, Tol tol) {
                // intersection with sets or points
                if (py::isinstance<ContSet>(S))
                    return Z.is_intersecting(con_zonotope_of(S), type, tol);
                return Z.is_intersecting(tensor_of(S, "the points"), type, tol);
            },
            py::arg("S"), py::arg("type") = "exact", py::arg("tol") = py::none(),
            "Whether Z and a set or points (n, N) intersect; 'exact' (default) or 'approx'")
        .def("split", py::overload_cast<>(&ConZonotope::split, py::const_),
             "One pair of halves of the set cut at the box center per coordinate")
        .def("split", py::overload_cast<int64_t>(&ConZonotope::split, py::const_), py::arg("N"),
             "The halves of the set cut at the center of the box in coordinate N")
        .def(
            "unite",
            [](const ConZonotope &Z, const py::object &S, const std::string &method,
               const std::optional<int> &order) {
                if (py::isinstance<py::list>(S) || py::isinstance<py::tuple>(S)) {
                    std::vector<ConZonotope> others;
                    for (const py::handle &o : S) others.push_back(con_zonotope_of(o));
                    return Z.unite(others, method, order);
                }
                if (py::isinstance<ContSet>(S)) return Z.unite(con_zonotope_of(S), method, order);
                // union
                return Z.unite(vector_of(S, "the point"), method, order);
            },
            py::arg("S"), py::arg("method") = "linprog", py::arg("order") = py::none(),
            "A set without constraints that contains the union with S (a set, a point or a list)")
        .def(
            "minkowski_difference",
            [](const ConZonotope &Z, const py::object &S2, const std::optional<std::string> &type) {
                if (py::isinstance<ContSet>(S2))
                    return object_of(Z.minkowski_difference(set_of(S2), type));
                return py::cast(Z.minkowski_difference(vector_of(S2, "the point")));
            },
            py::arg("S2"), py::arg("type") = py::none(),
            "The Minkowski difference {x | x + S2 in Z} with a zonotope, a box or a point")
        .def("compact", &ConZonotope::compact, py::arg("method") = py::none(),
             py::arg("tol") = py::none(),
             "Without the rows 0 = 0 and the generators that are zero and in no constraint")
        .def(
            "cart_prod",
            [](const ConZonotope &Z, const py::object &S) {
                if (py::isinstance<ContSet>(S)) return Z.cart_prod(con_zonotope_of(S));
                // Cartesian products
                return Z.cart_prod(vector_of(S, "the point"));
            },
            py::arg("S"),
            "The Cartesian product with a constrained zonotope, a zonotope, a box or a point")
        .def(
            "intersect",
            [](const ConZonotope &Z, const py::object &S) {
                return Z.intersect(con_zonotope_of(S));
            },
            py::arg("S"), "The intersection with a constrained zonotope, a zonotope or a box")
        .def(
            "con_intersect",
            [](const ConZonotope &Z, const ConZonotope &Z2, const py::object &M) {
                return Z.con_intersect(Z2, tensor_of(M, "M"));
            },
            py::arg("Z2"), py::arg("M"), "The points x of the set with M x in Z2")
        .def(
            "enclose",
            [](const ConZonotope &Z, const py::object &first, const py::object &second) {
                if (second.is_none()) return Z.enclose(py::cast<ConZonotope>(first));
                // enclose with a set or a point
                if (py::isinstance<ConZonotope>(second))
                    return Z.enclose(tensor_of(first, "M"), py::cast<ConZonotope>(second));
                return Z.enclose(tensor_of(first, "M"), vector_of(second, "t"));
            },
            py::arg("S"), py::arg("t") = py::none(),
            "Encloses the set and a linear image of it: enclose(Z2), or enclose(M, S) for M Z + S "
            "with a constrained zonotope or a point S")
        .def(
            "convex_hull",
            [](const ConZonotope &Z, const py::object &S, const std::string &method) {
                if (py::isinstance<py::list>(S)) {
                    std::vector<ConZonotope> others;
                    for (const py::handle &o : S) others.push_back(con_zonotope_of(o));
                    return Z.convex_hull(others);
                }
                if (py::isinstance<ContSet>(S)) return Z.convex_hull(con_zonotope_of(S), method);
                return Z.convex_hull(vector_of(S, "the point"), method);
            },
            py::arg("S"), py::arg("method") = "exact:Raghuraman",
            "The exact convex hull with a set, a point or a list of constrained zonotopes")
        // the cubic map
        .def(
            "cubic_map",
            [](const ConZonotope &Z, const py::list &T, const std::optional<Indices> &ind) {
                return Z.cubic_map(matrices_of(T), ind.value_or(Indices{}));
            },
            py::arg("T"), py::arg("ind") = py::none(),
            "An enclosure of z_i = sum_k x_k (x' T[i][k] x); ind[i] lists the k of output i")
        .def(
            "cubic_map",
            [](const ConZonotope &Z, const ConZonotope &Z2, const ConZonotope &Z3,
               const py::list &T, const std::optional<Indices> &ind) {
                return Z.cubic_map(Z2, Z3, matrices_of(T), ind.value_or(Indices{}));
            },
            py::arg("Z2"), py::arg("Z3"), py::arg("T"), py::arg("ind") = py::none(),
            "An enclosure of (x1' T[i] x2) x3 for x1 in Z, x2 in Z2, x3 in Z3")
        .def(
            "intersect_strip",
            [](const ConZonotope &Z, const py::object &C, const py::object &phi,
               const py::object &y, const py::object &method) {
                // the strip intersection
                const Tensor Ct = tensor_of(C, "C"), pt = vector_of(phi, "phi"),
                             yt = vector_of(y, "y");
                if (py::isinstance<py::str>(method))
                    return Z.intersect_strip(Ct, pt, yt, py::cast<std::string>(method));
                if (py::isinstance<py::dict>(method)) {
                    const py::dict d = py::cast<py::dict>(method);
                    const auto field = [&](const char *key) {
                        if (!d.contains(key))
                            CORA_FAIL(Error::wrong_field_value,
                                      std::string("intersect_strip: the observer needs the key '") +
                                          key + "'");
                        return tensor_of(d[key], key);
                    };
                    return Z.intersect_strip(Ct, pt, yt,
                                             {field("E"), field("F"), field("A"), field("C")});
                }
                // the strip intersection with a gain matrix
                return Z.intersect_strip(Ct, pt, yt, tensor_of(method, "the gain"));
            },
            py::arg("C"), py::arg("phi"), py::arg("y"), py::arg("method") = "normGen",
            "The intersection with |C x - y| <= phi; method: a name, a gain (n, k) or a dict with "
            "the observer matrices E, F, A, C ('wang-FRad')")
        .def_static(
            "approximate_bounds",
            [](const py::object &c, const py::object &G, const py::object &A, const py::object &b,
               const py::object &dr, int num_union_const, const std::string &method, int max_iter,
               bool batch_union) {
                ConZonotope::BoundsOptions options;
                options.method = method;
                options.max_iter = max_iter;
                options.batch_union = batch_union;
                const ConZonotope::Bounds r = ConZonotope::approximate_bounds(
                    vector_of(c, "c"), tensor_of(G, "G"), tensor_of(A, "A"), vector_of(b, "b"),
                    dr.is_none() ? std::nullopt : std::optional<Tensor>(vector_of(dr, "dr")),
                    num_union_const, options);
                // the bounds of a constrained zonotope
                return py::make_tuple(vector(r.l), vector(r.u), vector(r.bl), vector(r.bu));
            },
            py::arg("c"), py::arg("G"), py::arg("A"), py::arg("b"), py::arg("dr") = py::none(),
            py::arg("num_union_const") = 1, py::arg("method") = "fourier-motzkin",
            py::arg("max_iter") = 4, py::arg("batch_union") = true,
            "(l, u, bl, bu): bounds of c + G x + [-dr, dr], |x|_inf <= 1, A x <= b for a batch; "
            "the first num_union_const rows of A are a union");
}

// ZonoBundle --------------------------------------------------------------------------------------

/// The zonotopes of a Python list: Zonotope objects.
std::vector<Zonotope> zonotopes_of(const py::handle &list) {
    std::vector<Zonotope> out;
    for (const py::handle &o : py::reinterpret_borrow<py::sequence>(list)) {
        if (!py::isinstance<Zonotope>(o))
            CORA_FAIL(Error::wrong_input_in_constructor,
                      "ZonoBundle: the first input argument has to be a list of Zonotope objects.");
        out.push_back(py::cast<Zonotope>(o));
    }
    return out;
}

void bind_zono_bundle(py::module_ &m) {
    py::class_<ZonoBundle, ContSet>(m, "ZonoBundle",
                                    "The intersection of a list of zonotopes (a zonotope bundle).")
        .def(py::init([](const py::object &zonotopes) {
                 // a zonotope or a box is a bundle of one member
                 if (py::isinstance<Zonotope>(zonotopes))
                     return ZonoBundle::from(py::cast<Zonotope>(zonotopes));
                 if (py::isinstance<Interval>(zonotopes))
                     return ZonoBundle::from(py::cast<Interval>(zonotopes));
                 return ZonoBundle(zonotopes_of(zonotopes));
             }),
             py::arg("zonotopes"),
             "The intersection of a list of zonotopes (one dimension, at least one); a zonotope or "
             "a box gives a bundle of one member")
        .def_property_readonly(
            "zonotopes",
            [](const ZonoBundle &bundle) {
                py::list out;
                // the member zonotopes as a list
                for (const Zonotope &Z : bundle.zonotopes()) out.append(Z);
                return out;
            },
            "The member zonotopes")
        .def_property_readonly("parallel_sets", &ZonoBundle::parallel_sets,
                               "The number of member zonotopes")
        .def_static("empty", &ZonoBundle::empty, py::arg("n"),
                    "The empty set of n >= 1 dimensions: two zonotopes {0} and {e_1}")
        // the maps, sums and conversions of a bundle
        .def_static("origin", &ZonoBundle::origin, py::arg("n"), "The set {0} of n dimensions")
        .def_static(
            "generate_random",
            [](cora::Rng &rng, std::optional<int64_t> dimension,
               std::optional<int64_t> nr_zonotopes) {
                return ZonoBundle::generate_random(rng, dimension, nr_zonotopes);
            },
            py::arg("rng"), py::arg("dimension") = py::none(), py::arg("nr_zonotopes") = py::none(),
            "A random bundle of nr_zonotopes (default 2..5) zonotopes with a common point")
        .def("con_zonotope", &ZonoBundle::con_zonotope, "The same set as a constrained zonotope")
        // the other forms of the same set
        .def("zonotope", &ZonoBundle::zonotope, "The member of least volume")
        .def("polytope", &ZonoBundle::polytope, "The intersection of the polytopes of the members")
        // maps and sums
        .def(
            "linear_map",
            [](const ZonoBundle &Z, const py::object &M) {
                return Z.linear_map(tensor_of(M, "the matrix"));
            },
            py::arg("M"), "M Z for a matrix (..., k, n): every member mapped")
        // the sum with a point or a set
        .def(
            "plus",
            [](const ZonoBundle &Z, const py::object &S) {
                if (py::isinstance<Zonotope>(S)) return Z.plus(py::cast<Zonotope>(S));
                if (py::isinstance<Interval>(S)) return Z.plus(py::cast<Interval>(S));
                if (py::isinstance<ZonoBundle>(S)) return Z.plus(py::cast<ZonoBundle>(S));
                return Z.plus(vector_of(S, "the point"));
            },
            py::arg("S"), "The Minkowski sum with a bundle, a zonotope, a box or a point")
        .def(
            "quad_map",
            [](const ZonoBundle &Z, const py::list &Q) { return Z.quad_map(matrix_list_of(Q)); },
            py::arg("Q"), "An enclosure of (x' Q[i] x) for x in Z; Q[i] are (n, n) matrices")
        // fewer generators in every member
        .def(
            "reduce",
            [](const ZonoBundle &Z, int order, const std::string &method) {
                return Z.reduce(order, method);
            },
            py::arg("order") = 1, py::arg("method") = "girard",
            "Every member reduced to about order * dim generators (Girard)")
        // replacing a member, the type of the set, and the pieces of a split
        .def("replace", &ZonoBundle::replace, py::arg("index"), py::arg("Z"),
             "The bundle with member `index` (0-based) replaced by the zonotope Z")
        .def("representsa", &ZonoBundle::representsa, py::arg("type"), py::arg("tol") = py::none(),
             "Whether the bundle is also a set of the CORA type ('origin', 'interval', "
             "'emptySet', 'polytope', ...)")
        .def("split", py::overload_cast<>(&ZonoBundle::split, py::const_),
             "One pair of pieces cut at the middle of the box per coordinate")
        .def("split", py::overload_cast<int64_t>(&ZonoBundle::split, py::const_), py::arg("N"),
             "The pieces cut at the middle of coordinate N of the box around the bundle")
        // the union and random points
        .def(
            "unite",
            [](const ZonoBundle &Z, const py::object &S) {
                if (py::isinstance<Zonotope>(S)) return Z.unite(py::cast<Zonotope>(S));
                if (py::isinstance<Interval>(S)) return Z.unite(py::cast<Interval>(S));
                return Z.unite(py::cast<ZonoBundle>(S));
            },
            py::arg("S"), "A bundle that contains the union with a bundle, a zonotope or a box")
        // random points of either type
        .def(
            "rand_point",
            [](const ZonoBundle &Z, int64_t N, cora::Rng &rng, const std::string &type) {
                return matrix(Z.rand_point(N, rng, type));
            },
            py::arg("N"), py::arg("rng"), py::arg("type") = "standard",
            "N random points as columns (n, N); type 'standard' or 'extreme'")
        .def(
            "contains",
            [](const ZonoBundle &Z, const py::object &S, const std::string &method, Tol tol) {
                if (py::isinstance<Zonotope>(S))
                    return Z.contains(py::cast<Zonotope>(S), method, tol);
                if (py::isinstance<Interval>(S))
                    return Z.contains(py::cast<Interval>(S), method, tol);
                if (py::isinstance<ZonoBundle>(S))
                    return Z.contains(py::cast<ZonoBundle>(S), method, tol);
                // the points
                return Z.contains(tensor_of(S, "the points"), method, tol);
            },
            py::arg("S"), py::arg("method") = "exact", py::arg("tol") = py::none(),
            "Whether the bundle contains the points (n, N), a zonotope, a box or a bundle")
        .def(
            "is_intersecting",
            [](const ZonoBundle &Z, const py::object &S, const std::string &type, Tol tol) {
                if (py::isinstance<Zonotope>(S))
                    return Z.is_intersecting(py::cast<Zonotope>(S), type, tol);
                if (py::isinstance<Interval>(S))
                    return Z.is_intersecting(py::cast<Interval>(S), type, tol);
                if (py::isinstance<ZonoBundle>(S))
                    return Z.is_intersecting(py::cast<ZonoBundle>(S), type, tol);
                return Z.is_intersecting(tensor_of(S, "the points"), type, tol);
            },
            py::arg("S"), py::arg("type") = "exact", py::arg("tol") = py::none(),
            "Whether the bundle and the points, zonotope, box or bundle S intersect")
        .def("isequal", &ZonoBundle::isequal, py::arg("S"), py::arg("tol") = py::none(),
             "Whether the bundles are the same set, up to tol")
        // intersection with a zonotope or a box
        .def(
            "intersect",
            [](const ZonoBundle &Z, const py::object &S) {
                if (py::isinstance<Zonotope>(S)) return Z.intersect(py::cast<Zonotope>(S));
                if (py::isinstance<Interval>(S)) return Z.intersect(py::cast<Interval>(S));
                return Z.intersect(py::cast<ZonoBundle>(S));
            },
            py::arg("S"),
            "The intersection with a bundle, a zonotope or a box: the members of both")
        .def(
            "cart_prod",
            [](const ZonoBundle &Z, const py::object &S) {
                if (py::isinstance<Zonotope>(S)) return Z.cart_prod(py::cast<Zonotope>(S));
                if (py::isinstance<Interval>(S)) return Z.cart_prod(py::cast<Interval>(S));
                if (py::isinstance<ZonoBundle>(S)) return Z.cart_prod(py::cast<ZonoBundle>(S));
                return Z.cart_prod(vector_of(S, "the point"));
            },
            py::arg("S"), "The Cartesian product with a bundle, a zonotope, a box or a point")
        // enclose with a set
        .def(
            "enclose",
            [](const ZonoBundle &Z, const ContSet &S2) {
                return object_of(static_cast<const ContSet &>(Z).enclose(S2));
            },
            py::arg("S2"), "A bundle that encloses both, member by member")
        .def(
            "enclose",
            [](const ZonoBundle &Z, const py::object &M, const py::object &S) {
                if (py::isinstance<Zonotope>(S))
                    return Z.enclose(tensor_of(M, "the matrix"), py::cast<Zonotope>(S));
                return Z.enclose(tensor_of(M, "the matrix"), py::cast<ZonoBundle>(S));
            },
            py::arg("M"), py::arg("S"),
            "A bundle that encloses Z and M Z + S for a zonotope or a bundle S");
}

// Polytope ----------------------------------------------------------------------------------------

/// The polytope of the vertices (a matrix), the halfspaces (A, b) with the equalities (Ae, be) or
/// a box.
Polytope polytope_of(const py::object &first, const py::object &b, const py::object &Ae,
                     const py::object &be) {
    if (Ae.is_none() != be.is_none())
        CORA_FAIL(Error::wrong_input_in_constructor,
                  "Polytope: give the equality matrix and the offset together");
    if (py::isinstance<Interval>(first)) return Polytope(py::cast<Interval>(first));
    if (py::isinstance<EmptySet>(first)) return Polytope(py::cast<EmptySet>(first));
    if (py::isinstance<Fullspace>(first)) return Polytope(py::cast<Fullspace>(first));
    if (b.is_none()) return Polytope(tensor_of(first, "the vertices V"));
    if (Ae.is_none()) return Polytope(tensor_of(first, "A"), vector_of(b, "b"));
    return Polytope(tensor_of(first, "A"), vector_of(b, "b"), tensor_of(Ae, "Ae"),
                    vector_of(be, "be"));
}

/// A polytope from Python: a Polytope, or a box as one.
Polytope polytope_arg(const py::handle &o) {
    if (py::isinstance<Polytope>(o)) return py::cast<Polytope>(o);
    if (py::isinstance<Interval>(o)) return Polytope(py::cast<Interval>(o));
    CORA_FAIL(Error::wrong_value, "cora: expected a Polytope or an Interval");
}

void bind_polytope(py::module_ &m) {
    py::class_<Polytope, ContSet>(
        m, "Polytope", "The polytope {x : A x <= b, Ae x = be} or the hull of vertices V.")
        .def(py::init([](const py::object &first, const py::object &b, const py::object &Ae,
                         const py::object &be) { return polytope_of(first, b, Ae, be); }),
             py::arg("A"), py::arg("b") = py::none(), py::arg("Ae") = py::none(),
             py::arg("be") = py::none(),
             "Polytope(V): vertices (n, p); Polytope(A, b[, Ae, be]): halfspaces; Polytope(I): a "
             "box")
        .def_static("empty", &Polytope::empty, py::arg("n") = 0, "The empty set of n dimensions")
        .def_static("inf", &Polytope::inf, py::arg("n") = 0, "R^n")
        .def_static(
            "enclose_points",
            [](const py::object &points) {
                // the hull of the points
                return Polytope::enclose_points(tensor_of(points, "points"));
            },
            py::arg("points"), "The hull of the points (n, N), by its extreme points")
        .def_property_readonly(
            "A", [](const Polytope &P) { return matrix(P.A()); },
            "The inequality matrix (..., m, n)")
        .def_property_readonly(
            "b", [](const Polytope &P) { return vector(P.b()); }, "The inequality offset (..., m)")
        .def_property_readonly(
            "Ae", [](const Polytope &P) { return matrix(P.Ae()); },
            "The equality matrix (..., me, n)")
        .def_property_readonly(
            "be", [](const Polytope &P) { return vector(P.be()); }, "The equality offset (..., me)")
        .def_property_readonly(
            "V", [](const Polytope &P) { return matrix(P.V()); }, "The vertices (..., n, p)")
        .def(
            "constraints",
            [](const Polytope &P) {
                const Polytope::Constraints C = P.constraints();
                // the halfspaces as a tuple
                return py::make_tuple(matrix(C.A), vector(C.b), matrix(C.Ae), vector(C.be));
            },
            "(A, b, Ae, be): the halfspaces, from the vertices if need be")
        .def("box", &Polytope::box, "The polytope of the tightest box")
        .def(
            "center_by",
            [](const Polytope &P, const std::string &method) { return bounds(P.center(method)); },
            py::arg("method"),
            "The Chebyshev center ('chebyshev') or the mean of the vertices ('avg')")
        .def(
            "compact",
            [](const Polytope &P, const std::optional<std::string> &method, Tol tol) {
                return P.compact(method.value_or("all"), tol);
            },
            py::arg("method") = py::none(), py::arg("tol") = py::none(),
            "Without redundancy: all (default), A, Ae, aligned, zeros, AtoAe or V")
        .def(
            "con_zonotope",
            [](const Polytope &P, const std::string &method, const std::optional<Interval> &box) {
                // the polytope as a constrained zonotope
                return P.con_zonotope(method, box);
            },
            py::arg("method") = "exact:supportFunc", py::arg("box") = py::none(),
            "The polytope as a constrained zonotope: exact:supportFunc or exact:vertices")
        .def(
            "contains",
            [](const Polytope &P, const py::object &S, const std::string &method, Tol tol) {
                if (py::isinstance<ContSet>(S))
                    return P.contains(py::cast<const ContSet &>(S), method, tol);
                return P.contains(tensor_of(S, "the points"), method, tol);
            },
            py::arg("S"), py::arg("method") = "exact", py::arg("tol") = py::none(),
            "Whether the polytope contains the points (n, N) or a set")
        .def(
            "intersect",
            [](const Polytope &P, const py::object &S, Tol tol) {
                if (py::isinstance<ContSet>(S)) return P.intersect(polytope_arg(S));
                // intersection with a polytope, a box or a point
                return P.intersect(tensor_of(S, "the point"), tol);
            },
            py::arg("S"), py::arg("tol") = py::none(),
            "The intersection with a polytope, a box or a point")
        .def(
            "cart_prod",
            [](const Polytope &P, const py::object &S) {
                if (py::isinstance<ContSet>(S)) return P.cart_prod(polytope_arg(S));
                return P.cart_prod(vector_of(S, "the point"));
            },
            py::arg("S"), "The Cartesian product with a polytope, a box or a point")
        .def(
            "convex_hull",
            [](const Polytope &P, const py::object &S) {
                if (py::isinstance<ContSet>(S)) return P.convex_hull(polytope_arg(S));
                return P.convex_hull(vector_of(S, "the point"));
            },
            py::arg("S"), "The convex hull with a polytope, a box or a point")
        // enclose with a set
        .def(
            "enclose",
            [](const Polytope &P, const ContSet &S2) {
                return object_of(static_cast<const ContSet &>(P).enclose(S2));
            },
            py::arg("P2"), "The convex hull of the polytope and P2")
        .def(
            "hausdorff_distance",
            [](const Polytope &P, const py::object &S) {
                if (py::isinstance<Interval>(S)) return P.hausdorff_distance(py::cast<Interval>(S));
                if (py::isinstance<Polytope>(S)) return P.hausdorff_distance(py::cast<Polytope>(S));
                return P.hausdorff_distance(tensor_of(S, "the points"));
            },
            py::arg("S"), "The Hausdorff distance to a polytope, a box or the points (n, N)")
        // the second batch: the statics, the queries, the maps and the conversions
        .def_static("origin", &Polytope::origin, py::arg("n"), "The origin of R^n as a polytope")
        .def_static(
            "generate_random",
            [](cora::Rng &rng, std::optional<int64_t> dimension,
               std::optional<int64_t> nr_constraints, std::optional<bool> is_degenerate,
               std::optional<bool> is_bounded) {
                return Polytope::generate_random(rng, dimension, nr_constraints, is_degenerate,
                                                 is_bounded);
            },
            py::arg("rng"), py::arg("dimension") = py::none(),
            py::arg("nr_constraints") = py::none(), py::arg("is_degenerate") = py::none(),
            py::arg("is_bounded") = py::none(),
            "A random polytope; unset properties are drawn from the generator")
        // the queries
        .def("representsa", &Polytope::representsa, py::arg("type"), py::arg("tol") = py::none(),
             "Whether the polytope is also a set of the CORA type ('origin', 'point', "
             "'emptySet', 'fullspace', 'interval', 'zonotope', ...)")
        .def("volume", &Polytope::volume, "The volume (0 if it is not full-dimensional)")
        .def(
            "is_full_dim_subspace",
            [](const Polytope &P, Tol tol) {
                const auto [full, subspace] = P.is_full_dim_subspace(tol);
                return py::make_tuple(full, matrix(subspace));
            },
            py::arg("tol") = py::none(),
            "(full, subspace): whether the polytope is full-dimensional, else an orthonormal "
            "basis of the subspace it spans")
        // the comparisons with sets and points
        .def(
            "isequal",
            [](const Polytope &P, const py::object &S, Tol tol) {
                if (py::isinstance<Polytope>(S)) return P.isequal(py::cast<Polytope>(S), tol);
                if (py::isinstance<ContSet>(S)) return P.isequal(py::cast<const ContSet &>(S), tol);
                return P.isequal(tensor_of(S, "the point"), tol);
            },
            py::arg("S"), py::arg("tol") = py::none(),
            "Whether the polytope equals a set or a point")
        // the intersection test by the class of the operand
        .def(
            "is_intersecting",
            [](const Polytope &P, const py::object &S, const std::string &type, Tol tol) {
                if (py::isinstance<Polytope>(S))
                    return P.is_intersecting(py::cast<Polytope>(S), type, tol);
                if (py::isinstance<Interval>(S))
                    return P.is_intersecting(py::cast<Interval>(S), type, tol);
                if (py::isinstance<Zonotope>(S))
                    return P.is_intersecting(py::cast<Zonotope>(S), type, tol);
                if (py::isinstance<ConZonotope>(S))
                    return P.is_intersecting(py::cast<ConZonotope>(S), type, tol);
                if (py::isinstance<ZonoBundle>(S))
                    return P.is_intersecting(py::cast<ZonoBundle>(S), type, tol);
                return P.is_intersecting(tensor_of(S, "the point"), type, tol);
            },
            py::arg("S"), py::arg("type") = "exact", py::arg("tol") = py::none(),
            "Whether the polytope and a set or a point intersect; type 'exact' or 'approx'")
        // containment in a set, as a method and as the operator
        .def("less_equal", &Polytope::less_equal, py::arg("S"), py::arg("tol") = py::none(),
             "Whether the polytope is a subset of S")
        .def(
            "__le__", [](const Polytope &P, const ContSet &S) { return P.less_equal(S); },
            py::arg("S"), "Whether the polytope is a subset of S")
        // the distance and the sets built from the set
        .def(
            "distance",
            [](const Polytope &P, const py::object &S) -> py::object {
                if (py::isinstance<Polytope>(S)) return py::cast(P.distance(py::cast<Polytope>(S)));
                return vector(P.distance(tensor_of(S, "the points")));
            },
            py::arg("S"), "The distance to a polytope or the signed distance to points (n, N)")
        // the sets built from the set
        .def(
            "unite",
            [](const Polytope &P, const py::object &S) {
                if (py::isinstance<Polytope>(S)) return P.unite(py::cast<Polytope>(S));
                if (py::isinstance<Interval>(S)) return P.unite(py::cast<Interval>(S));
                return P.unite(tensor_of(S, "the point"));
            },
            py::arg("S"), "The convex hull of the union with a polytope, a box or points")
        // the normal forms and the reductions
        .def("complement", &Polytope::complement, "The union of the complementary halfspaces")
        .def("normalize_constraints", &Polytope::normalize_constraints, py::arg("type") = "b",
             "Rows of unit norm ('A') or of unit offset ('b')")
        .def("reduce_over_domain", &Polytope::reduce_over_domain, py::arg("dom"),
             py::arg("type") = "outer", "Drops the constraints that are inactive on the box dom")
        .def(
            "reduce",
            [](const Polytope &P, int order, std::optional<std::reference_wrapper<cora::Rng>> rng,
               const std::string &method) {
                // without a generator, a fixed seed gives the same directions every time
                cora::Rng fallback(0);
                return P.reduce(order, rng ? rng->get() : fallback, method);
            },
            py::arg("order") = 1, py::arg("rng") = py::none(), py::arg("method") = "rand",
            "An enclosing polytope with at most order * dim constraints")
        .def(
            "set_difference",
            [](const Polytope &P, const Polytope &S) {
                py::list parts;
                // the pieces of the difference
                for (const Polytope &part : P.set_difference(S)) parts.append(part);
                return parts;
            },
            py::arg("S"), "Polytopes whose union is the set difference (CORA's mldivide)")
        // the conversions and the samples
        .def("zonotope", &Polytope::zonotope, py::arg("method") = "outer",
             "A zonotope that encloses the polytope")
        .def("zono_bundle", &Polytope::zono_bundle, "The polytope as a zonotope bundle")
        .def("ellipsoid", &Polytope::ellipsoid, py::arg("method") = "outer",
             "An ellipsoid that encloses the polytope")
        .def(
            "rand_point",
            [](const Polytope &P, int64_t N, cora::Rng &rng, const std::string &type) {
                return matrix(P.rand_point(N, rng, type));
            },
            py::arg("N"), py::arg("rng"), py::arg("type") = "standard",
            "N random points as columns (n, N); type 'standard' or 'extreme'");
}

// Ellipsoid ---------------------------------------------------------------------------------------

/// The measurement model of the method of Liu from a Python dict with the keys V and bar_V.
Ellipsoid::StripMeasurement strip_measurement_of(const py::handle &o) {
    if (!py::isinstance<py::dict>(o))
        CORA_FAIL(Error::wrong_value, "intersect_strip: the method 'Liu2016' takes a dict with "
                                      "the matrices V and bar_V as the width");
    const py::dict d = py::cast<py::dict>(o);
    for (const char *key : {"V", "bar_V"})
        if (!d.contains(key))
            CORA_FAIL(Error::wrong_field_value,
                      std::string("intersect_strip: the model needs the key '") + key + "'");
    return {tensor_of(d["V"], "V"), tensor_of(d["bar_V"], "bar_V")};
}

/// An ellipsoid from a Python value: the ellipsoid itself.
Ellipsoid ellipsoid_of(const py::handle &o) { return py::cast<Ellipsoid>(o); }

void bind_ellipsoid(py::module_ &m) {
    py::class_<Ellipsoid, ContSet>(m, "Ellipsoid",
                                   "The ellipsoid {x : (x - q)' Q^+ (x - q) <= 1} of the shape "
                                   "matrix Q and the center q.")
        .def(py::init<const Interval &, const std::string &>(), py::arg("I"),
             py::arg("mode") = "outer", "An ellipsoid that holds a box; mode 'outer'")
        .def(py::init<const Zonotope &, const std::string &>(), py::arg("Z"),
             py::arg("mode") = "outer:norm_bnd",
             "An ellipsoid that holds a zonotope of n independent generators")
        .def(py::init([](const py::object &Q, const py::object &q, Tol tol) {
                 if (q.is_none()) return Ellipsoid(tensor_of(Q, "Q"), tol);
                 // a shape matrix without a center is centered at 0
                 return Ellipsoid(tensor_of(Q, "Q"), vector_of(q, "the center"), tol);
             }),
             py::arg("Q"), py::arg("q") = py::none(), py::arg("tol") = py::none(),
             "shape matrix Q (..., n, n), center q (..., n), default 0; tol bounds the asymmetry "
             "and the negative eigenvalues of Q")
        .def_static("empty", &Ellipsoid::empty, py::arg("n") = 0,
                    "The empty ellipsoid of n dimensions")
        .def_static("origin", &Ellipsoid::origin, py::arg("n"),
                    "The ellipsoid that is the origin of R^n")
        // a random ellipsoid
        .def_static(
            "generate_random",
            [](cora::Rng &rng, const std::optional<int64_t> &dimension, bool is_degenerate,
               const py::object &center) {
                return Ellipsoid::generate_random(rng, dimension, is_degenerate,
                                                  optional_vector(center, "center"));
            },
            py::arg("rng"), py::arg("dimension") = py::none(), py::arg("is_degenerate") = false,
            py::arg("center") = py::none(),
            "A random ellipsoid: the dimension (default that of center, else random in 1..30), "
            "whether Q is singular, and the center")
        .def(
            "rand_point",
            [](const Ellipsoid &E, int64_t N, cora::Rng &rng, const std::string &type) {
                return matrix(E.rand_point(N, rng, type));
            },
            py::arg("N"), py::arg("rng"), py::arg("type") = "standard",
            "N random points as columns (n, N); type 'standard', 'uniform' or 'extreme'")
        // small queries: NaN, semi-axes, volume and equality
        .def("isnan", &Ellipsoid::isnan, "Whether the ellipsoid holds NaN: always False")
        .def(
            "radius", [](const Ellipsoid &E, int64_t i) { return vector(E.radius(i)); },
            py::arg("i") = 1, "The semi-axes: the roots of the i largest eigenvalues of Q")
        .def(
            "volume", [](const Ellipsoid &E) { return scalar(E.volume()); }, "The volume")
        .def(
            "isequal",
            [](const Ellipsoid &E, const Ellipsoid &S, Tol tol) { return E.isequal(S, tol); },
            py::arg("S"), py::arg("tol") = py::none(),
            "Whether the ellipsoids are equal up to tol, or both empty")
        // the distance of points, the intersection tests and the sums
        .def(
            "distance",
            [](const Ellipsoid &E, const py::object &p) {
                return matrix(E.distance(tensor_of(p, "the points")));
            },
            py::arg("p"), "(x - q)' Q^-1 (x - q) - 1 for the points p (n, N): negative inside")
        // meeting points or an ellipsoid
        .def(
            "is_intersecting",
            [](const Ellipsoid &E, const py::object &S, Tol tol) {
                if (py::isinstance<Ellipsoid>(S)) return E.is_intersecting(ellipsoid_of(S), tol);
                return E.is_intersecting(tensor_of(S, "the points"), tol);
            },
            py::arg("S"), py::arg("tol") = py::none(),
            "Whether the ellipsoid meets an ellipsoid (one a point) or contains the points (n, N)")
        // the Minkowski sum
        .def(
            "plus",
            [](const Ellipsoid &E, const py::object &S, const std::string &mode) {
                if (py::isinstance<Ellipsoid>(S)) return E.plus(ellipsoid_of(S), mode);
                return E.plus(vector_of(S, "the vector"));
            },
            py::arg("S"), py::arg("mode") = "outer:halder",
            "The Minkowski sum with a vector or an ellipsoid (the enclosure of Halder)")
        // differences and unions
        .def(
            "mink_diff",
            [](const Ellipsoid &E, const py::object &S) {
                if (py::isinstance<Ellipsoid>(S)) return E.mink_diff(ellipsoid_of(S));
                return E.mink_diff(vector_of(S, "the vector"));
            },
            py::arg("S"), "The Minkowski difference with a vector or an ellipsoid")
        .def(
            "unite",
            [](const Ellipsoid &E, const py::object &S, const std::string &mode) {
                if (py::isinstance<Ellipsoid>(S)) return E.unite(ellipsoid_of(S), mode);
                return E.unite(vector_of(S, "the point"), mode);
            },
            py::arg("S"), py::arg("mode") = "outer",
            "An ellipsoid that holds the union with an ellipsoid or a point")
        .def("convex_hull", &Ellipsoid::convex_hull, py::arg("S"),
             "An ellipsoid that holds the convex hull with S")
        // the parallelotope around or inside the ellipsoid
        .def("zonotope", &Ellipsoid::zonotope, py::arg("mode") = "outer:box",
             py::arg("number_of_generators") = py::none(),
             "A zonotope that holds ('outer:box') or lies in ('inner:box') the ellipsoid")
        .def_property_readonly(
            "Q", [](const Ellipsoid &E) { return matrix(E.Q()); }, "The shape matrix Q (..., n, n)")
        .def_property_readonly(
            "q", [](const Ellipsoid &E) { return vector(E.q()); }, "The center q (..., n)")
        .def_property_readonly("tol", &Ellipsoid::tol, "The tolerance of the object")
        .def("is_bounded", &Ellipsoid::is_bounded, "Always true")
        .def("is_full_dim", &Ellipsoid::is_full_dim, py::arg("tol") = py::none(),
             "Whether the shape matrix has full rank; tol replaces the tolerance of the object")
        .def(
            "generators", [](const Ellipsoid &E) { return matrix(E.generators()); },
            "A matrix G with G G' = Q from the singular value decomposition")
        // the generators and the norm of the ellipsoid
        .def(
            "ellipsoid_norm",
            [](const Ellipsoid &E, const py::object &p) {
                return scalar(E.ellipsoid_norm(vector_of(p, "p")));
            },
            py::arg("p"), "The norm of p whose unit ball is the ellipsoid shifted to the origin")
        .def("is_bigger", &Ellipsoid::is_bigger, py::arg("E2"), "Whether E2 - q2 lies in E - q")
        .def(
            "is_bad_dir",
            [](const Ellipsoid &E, const Ellipsoid &E2, const py::object &L) {
                return E.is_bad_dir(E2, tensor_of(L, "L"));
            },
            py::arg("E2"), py::arg("L"),
            "For every column of L whether it is a bad direction of the Minkowski difference")
        .def_static(
            "enclose_points",
            [](const py::object &points, const std::string &method) {
                // an ellipsoid around points
                return Ellipsoid::enclose_points(tensor_of(points, "points"), method);
            },
            py::arg("points"), py::arg("method") = "cov",
            "An ellipsoid that contains the points (n, p); method 'cov'")
        .def(
            "intersect_strip",
            [](const Ellipsoid &E, const py::object &C, const py::object &phi, const py::object &y,
               double sigma_sq_prev, const std::optional<std::string> &method) {
                const bool liu = py::isinstance<py::dict>(phi);
                if (method && *method != (liu ? "Liu2016" : "Gollamudi1996"))
                    CORA_FAIL(Error::wrong_value,
                              "intersect_strip: the method '" + *method +
                                  "' does not go with this width (a number: 'Gollamudi1996', "
                                  "a dict of V and bar_V: 'Liu2016')");
                const Tensor Ct = tensor_of(C, "C"), yt = vector_of(y, "y");
                const auto result =
                    liu ? E.intersect_strip(Ct, strip_measurement_of(phi), yt, sigma_sq_prev)
                        : E.intersect_strip(Ct, py::cast<double>(phi), yt, sigma_sq_prev);
                // the strip intersection returns the ellipsoid and its sigma
                return py::make_tuple(result.first, result.second);
            },
            py::arg("C"), py::arg("phi"), py::arg("y"), py::arg("sigma_sq_prev") = 0.0,
            py::arg("method") = py::none(),
            "(E2, sigma_sq): an ellipsoid with the intersection with the strip |C x - y| <= phi "
            "(method 'Gollamudi1996', a number phi) or the measurement model {V, bar_V} "
            "(method 'Liu2016')")
        .def(
            "intersect",
            [](const Ellipsoid &E, const py::object &S, const std::string &mode) {
                if (py::isinstance<Ellipsoid>(S)) return E.intersect(ellipsoid_of(S), mode);
                return E.intersect(vector_of(S, "the point"));
            },
            py::arg("S"), py::arg("mode") = "outer",
            "The intersection with an ellipsoid or a point (an ellipsoid with Q = 0 or empty)")
        .def(
            "cart_prod",
            [](const Ellipsoid &E, const py::object &S, const std::string &mode) {
                if (py::isinstance<Ellipsoid>(S)) return E.cart_prod(ellipsoid_of(S), mode);
                // Cartesian product with an ellipsoid or a point
                return E.cart_prod(vector_of(S, "the point"), mode);
            },
            py::arg("S"), py::arg("mode") = "outer",
            "The Cartesian product with an ellipsoid (of the bounding boxes) or a point (exact)")
        .def(
            "contains",
            [](const Ellipsoid &E, const py::object &S, const std::string &method, Tol tol) {
                if (py::isinstance<Zonotope>(S))
                    return E.contains(py::cast<Zonotope>(S), method, tol);
                if (py::isinstance<Interval>(S))
                    return E.contains(py::cast<Interval>(S), method, tol);
                if (py::isinstance<Ellipsoid>(S)) return E.contains(ellipsoid_of(S), method, tol);
                return E.contains(tensor_of(S, "the points"), method, tol);
            },
            py::arg("S"), py::arg("method") = "exact", py::arg("tol") = py::none(),
            "Whether the ellipsoid contains the points (n, N), a zonotope, a box or an ellipsoid")
        .def(
            "enclose",
            [](const Ellipsoid &E, const ContSet &S2) {
                // enclose with an ellipsoid or a set
                if (S2.class_id() == ClassId::ellipsoid)
                    return object_of(Set(E.enclose(static_cast<const Ellipsoid &>(S2))));
                return object_of(static_cast<const ContSet &>(E).enclose(S2));
            },
            py::arg("S2"), "The outer union of E and S2; needs an SDP solver")
        .def(
            "enclose",
            [](const Ellipsoid &E, const py::object &M, const Ellipsoid &Eplus) {
                return E.enclose(tensor_of(M, "the matrix"), Eplus);
            },
            py::arg("M"), py::arg("Eplus"),
            "The outer union of E and M Eplus (CORA's code); needs an SDP solver");
}

// Capsule -----------------------------------------------------------------------------------------

/// The radius of a capsule from Python: a number for every member, or an array of the batch.
Tensor capsule_radius_of(const py::object &r) { return vector_of(r, "the radius").expand_dims(-1); }

void bind_capsule(py::module_ &m) {
    py::class_<Capsule, ContSet>(m, "Capsule",
                                 "The capsule: the points within r of the segment c - g .. c + g.")
        .def(py::init<const Interval &>(), py::arg("I"), "A capsule that holds a box")
        .def(py::init<const Zonotope &>(), py::arg("Z"), "A capsule that holds a zonotope")
        .def(py::init([](const py::object &c, const py::object &g, const py::object &r) {
                 const Tensor center = vector_of(c, "the center");
                 const Tensor generator =
                     g.is_none() ? center.zeros_like() : vector_of(g, "the generator");
                 if (r.is_none()) return Capsule(center, generator);
                 if (is_number(r)) return Capsule(center, generator, py::cast<double>(r));
                 // the radius is absent, a number or a vector
                 return Capsule(center, generator, capsule_radius_of(r));
             }),
             py::arg("c"), py::arg("g") = py::none(), py::arg("r") = py::none(),
             "center c (..., n), generator g (..., n; default 0) and radius r (default 0)")
        .def_static("empty", &Capsule::empty, py::arg("n") = 0,
                    "The empty capsule of n dimensions: the radius -1")
        .def_static("origin", &Capsule::origin, py::arg("n"),
                    "The capsule that is the origin of R^n")
        // a random capsule
        .def_static(
            "generate_random",
            [](cora::Rng &rng, const std::optional<int64_t> &dimension, const py::object &center,
               const std::optional<double> &radius) {
                return Capsule::generate_random(rng, dimension, optional_vector(center, "center"),
                                                radius);
            },
            py::arg("rng"), py::arg("dimension") = py::none(), py::arg("center") = py::none(),
            py::arg("radius") = py::none(),
            "A random capsule: the dimension (default that of center, else random in 1..10), the "
            "center and the radius (default random in [0, 5])")
        .def(
            "rand_point",
            [](const Capsule &C, int64_t N, cora::Rng &rng, const std::string &type) {
                return matrix(C.rand_point(N, rng, type));
            },
            py::arg("N"), py::arg("rng"), py::arg("type") = "standard",
            "N random points as columns (..., n, N); type 'standard' or 'extreme'")
        // small queries: NaN, radius, volume and the type
        .def("isnan", &Capsule::isnan, "Whether the capsule holds NaN: always False")
        .def(
            "radius", [](const Capsule &C) { return scalar(C.radius()); },
            "The radius of a ball around the center that holds the capsule: |g| + r")
        // the volume and the type
        .def(
            "volume", [](const Capsule &C) { return scalar(C.volume()); }, "The volume")
        .def("representsa", &Capsule::representsa, py::arg("type"), py::arg("tol") = py::none(),
             "Whether the capsule is also a set of the CORA type ('point', 'zonotope', ...)")
        .def(
            "plus",
            [](const Capsule &C, const py::object &S) {
                if (py::isinstance<Capsule>(S)) return C.plus(py::cast<Capsule>(S));
                if (py::isinstance<Interval>(S)) return C.plus(py::cast<Interval>(S));
                if (py::isinstance<Zonotope>(S)) return C.plus(py::cast<Zonotope>(S));
                return C.plus(vector_of(S, "the vector"));
            },
            py::arg("S"), "The Minkowski sum with a capsule, a box, a zonotope or a vector")
        // conversions
        .def("zonotope", &Capsule::zonotope, "A zonotope that holds the capsule")
        .def("ellipsoid", &Capsule::ellipsoid, py::arg("method") = "exact",
             "The ellipsoid of a capsule without generator")
        .def("polytope", &Capsule::polytope, py::arg("method") = "exact",
             "The polytope of the capsule: exact for a segment, 'outer' for the rest")
        .def_static(
            "enclose_points",
            [](const py::object &points) {
                return Capsule::enclose_points(tensor_of(points, "the points"));
            },
            py::arg("points"),
            "A capsule that contains the points (..., n, N): mean, farthest point, largest "
            "distance from the axis")
        .def_property_readonly(
            "c", [](const Capsule &C) { return vector(C.c()); }, "The stored center c (..., n)")
        // the stored center, generator and radius of the capsule
        .def_property_readonly(
            "generator", [](const Capsule &C) { return vector(C.generator()); },
            "The generator g (..., n)")
        .def_property_readonly(
            "r", [](const Capsule &C) { return scalar(C.r()); },
            "The radius r (-1 for the empty capsule)")
        .def("is_empty", &Capsule::is_empty,
             "Whether the capsule is empty (of a batch: every member)")
        .def(
            "isequal",
            [](const Capsule &C, const Capsule &S, Tol tol) { return C.isequal(S, tol); },
            py::arg("S"), py::arg("tol") = py::none(),
            "Whether the capsules are equal up to tol, or both empty")
        .def(
            "contains",
            [](const Capsule &C, const py::object &S, Tol tol) {
                if (py::isinstance<Capsule>(S)) return C.contains(py::cast<Capsule>(S), tol);
                if (py::isinstance<Zonotope>(S)) return C.contains(py::cast<Zonotope>(S), tol);
                if (py::isinstance<Interval>(S)) return C.contains(py::cast<Interval>(S), tol);
                // containment of points or sets
                return C.contains(tensor_of(S, "the points"), tol);
            },
            py::arg("S"), py::arg("tol") = py::none(),
            "Whether the capsule contains the points (n, N), a capsule, a zonotope or a box")
        .def(
            "is_intersecting",
            [](const Capsule &C, const py::object &S, Tol tol) {
                if (py::isinstance<Capsule>(S)) return C.is_intersecting(py::cast<Capsule>(S), tol);
                return C.is_intersecting(tensor_of(S, "the points"), tol);
            },
            py::arg("S"), py::arg("tol") = py::none(),
            "Whether the capsule meets a capsule or contains the points (n, N), up to tol")
        .def("convex_hull", &Capsule::convex_hull, py::arg("S"),
             "A capsule that encloses the capsule and S");
}

// PolyZonotope ------------------------------------------------------------------------------------

/// The exponent matrix E (p, h) of Python, as one column of p exponents per generator.
Exponents exponents_from_python(const py::object &E, std::size_t generators) {
    if (E.is_none()) {
        // CORA's default: the identity, a factor per generator
        Exponents out(generators, std::vector<int>(generators, 0));
        for (std::size_t j = 0; j < generators; ++j) out[j][j] = 1;
        return out;
    }
    const py::object rows = py::hasattr(E, "tolist") ? E.attr("tolist")() : py::object(E);
    const auto matrix = py::cast<std::vector<std::vector<int64_t>>>(rows);
    Exponents out(generators);
    for (std::size_t j = 0; j < generators; ++j)
        for (const auto &row : matrix) {
            if (row.size() != generators)
                CORA_FAIL(Error::wrong_input_in_constructor,
                          "PolyZonotope: E needs one column per generator of G");
            out[j].push_back(int(row[j]));
        }
    return out;
}

/// The tensor of an optional generator matrix: none is a matrix without columns.
Tensor generators_from_python(const py::object &G, const Tensor &c, const char *what) {
    // a missing matrix has the rows of the center and no column
    return G.is_none() ? c.zeros_like({c.shape()[0], 0}) : tensor_of(G, what);
}

void bind_poly_zonotope(py::module_ &m) {
    py::class_<PolyZonotope, ContSet>(
        m, "PolyZonotope",
        "The polynomial zonotope c + sum_j (prod_i a_i^E[i,j]) G[:,j] "
        "+ GI b with factors a_i, b in [-1, 1].")
        .def(py::init<const Zonotope &>(), py::arg("Z"), "A zonotope: its generators are dependent")
        .def(py::init<const Interval &>(), py::arg("I"), "A box: a dependent generator per axis")
        .def(py::init([](const py::object &c, const py::object &G, const py::object &GI,
                         const py::object &E, const py::object &id) {
                 const Tensor center = vector_of(c, "the center");
                 const Tensor dependent = generators_from_python(G, center, "G");
                 const auto h = std::size_t(dependent.shape()[1]);
                 // the exponents, and the identifiers 1..p unless they are given
                 Exponents exps = exponents_from_python(E, h);
                 std::vector<int64_t> ids;
                 if (id.is_none()) {
                     const std::size_t p = exps.empty() ? 0 : exps.front().size();
                     for (std::size_t i = 0; i < p; ++i) ids.push_back(int64_t(i) + 1);
                 } else {
                     ids = py::cast<std::vector<int64_t>>(id);
                 }
                 return PolyZonotope(center, dependent, generators_from_python(GI, center, "GI"),
                                     std::move(exps), std::move(ids));
             }),
             py::arg("c"), py::arg("G") = py::none(), py::arg("GI") = py::none(),
             py::arg("E") = py::none(), py::arg("id") = py::none(),
             "center c (n), dependent generators G (n, h), independent generators GI (n, q), "
             "exponents E (p, h; default the identity) and identifiers id (p; default 1..p)")
        // the stored parts as arrays
        .def_property_readonly(
            "c", [](const PolyZonotope &P) { return vector(P.center()); }, "The center c (n)")
        .def_property_readonly(
            "G", [](const PolyZonotope &P) { return matrix(P.generators()); },
            "The dependent generators G (n, h)")
        .def_property_readonly(
            "GI", [](const PolyZonotope &P) { return matrix(P.independent_generators()); },
            "The independent generators GI (n, q)")
        // the exponents as the matrix E (p, h) of CORA
        .def_property_readonly(
            "E",
            [](const PolyZonotope &P) {
                const auto p = P.num_factors(), h = P.num_dependent();
                py::array_t<int64_t> out({p, h});
                auto view = out.mutable_unchecked<2>();
                for (int64_t i = 0; i < p; ++i)
                    for (int64_t j = 0; j < h; ++j)
                        view(i, j) = P.exponents()[std::size_t(j)][std::size_t(i)];
                return out;
            },
            "The exponent matrix E (p, h)")
        .def_property_readonly(
            "id", [](const PolyZonotope &P) { return P.ids(); }, "The identifiers (p)")
        // the conversion, the sums and the maps
        .def("zonotope", &PolyZonotope::zonotope, "A zonotope that holds the set")
        .def(
            "plus",
            [](const PolyZonotope &P, const py::object &S) {
                if (py::isinstance<PolyZonotope>(S)) return P.plus(py::cast<PolyZonotope>(S));
                if (py::isinstance<Zonotope>(S)) return P.plus(py::cast<Zonotope>(S));
                if (py::isinstance<Interval>(S)) return P.plus(py::cast<Interval>(S));
                return P.plus(vector_of(S, "the point"));
            },
            py::arg("S"),
            "The sum with a zonotope, a box or a point (they join the independent part) or with a "
            "polynomial zonotope (on factors of its own)")
        // the sum with shared factors, and the maps
        .def("exact_plus", &PolyZonotope::exact_plus, py::arg("S"),
             "The sum in which equal identifiers are one factor")
        .def(
            "linear_map",
            [](const PolyZonotope &P, const py::object &M) {
                if (py::isinstance<IntervalMatrix>(M))
                    return P.linear_map(py::cast<IntervalMatrix>(M).interval());
                if (py::isinstance<Interval>(M)) return P.linear_map(py::cast<Interval>(M));
                return P.linear_map(tensor_of(M, "the matrix"));
            },
            py::arg("M"), "M * P for a matrix, or [M] * P for an interval matrix")
        // the hull, the product and the reduction
        .def(
            "enclose",
            [](const PolyZonotope &P, const ContSet &S) {
                return P.enclose(dynamic_cast<const PolyZonotope &>(S));
            },
            py::arg("S"),
            "A polynomial zonotope that holds the segments between the points of both")
        .def("cart_prod", &PolyZonotope::cart_prod, py::arg("Z"),
             "The Cartesian product with a zonotope (its generators are independent)")
        .def("reduce", &PolyZonotope::reduce, py::arg("order") = 1,
             "A set with at most order * dim generators that holds the set (Girard)")
        // the quadratic map
        .def(
            "quad_map",
            [](const PolyZonotope &P, const std::vector<py::object> &Q) {
                // the matrices Q[i] of the map
                std::vector<Tensor> matrices;
                matrices.reserve(Q.size());
                for (const py::object &q : Q) matrices.push_back(tensor_of(q, "a matrix of Q"));
                return P.quad_map(matrices);
            },
            py::arg("Q"), "The image under z_i = x' Q[i] x")
        // the canonical form
        .def(
            "compact",
            [](const PolyZonotope &P, const std::optional<std::string> &method, Tol) {
                if (method && *method != "all")
                    CORA_FAIL(Error::no_specific_alg, "compact: a polynomial zonotope has the "
                                                      "method 'all' only, got '" +
                                                          *method + "'");
                return P.compact();
            },
            py::arg("method") = py::none(), py::arg("tol") = py::none(),
            "The set without zero generators and unused factors (method 'all')")
        // the halves along a dependent factor
        .def("split_dep_factor", &PolyZonotope::split_dep_factor, py::arg("id"),
             "The halves along the factor with the identifier id: alpha becomes (alpha +- 1) / 2")
        .def("split_longest_gen", &PolyZonotope::split_longest_gen,
             "The halves along the factor of the longest dependent generator")
        // CORA's approximate test: False is certain
        .def(
            "is_intersecting",
            [](const PolyZonotope &P, const Polytope &S, Tol tol) {
                return P.is_intersecting(S, tol);
            },
            py::arg("S"), py::arg("tol") = py::none(),
            "False if the set certainly misses the polytope, else True");
    m.attr("PolyZonotope").attr("__array_ufunc__") = py::none();
}

void bind_operators(py::module_ &m) {
    const py::object Z = m.attr("Zonotope"), I = m.attr("Interval"), E = m.attr("EmptySet"),
                     F = m.attr("Fullspace");
    const py::object CZ = m.attr("ConZonotope");
    CZ.attr("__array_ufunc__") = py::none(); // numpy defers to its operators
    m.attr("Polytope").attr("__array_ufunc__") = py::none();
    m.attr("Ellipsoid").attr("__array_ufunc__") = py::none();
    m.attr("Capsule").attr("__array_ufunc__") = py::none();
    Z.attr("__array_ufunc__") = py::none(); // numpy defers to the operators below
    I.attr("__array_ufunc__") = py::none();
    E.attr("__array_ufunc__") = py::none();
    F.attr("__array_ufunc__") = py::none();
    const auto method = [](const py::object &cls, const char *name, auto f) {
        cls.attr(name) = py::cpp_function(f, py::is_method(cls));
    };

    // Polytope: a point shifts it, a set adds, a matrix maps it
    const py::object P = m.attr("Polytope");
    const auto polytope_sum = [](const Polytope &A, const py::object &o) -> py::object {
        if (py::isinstance<Polytope>(o)) return py::cast(A.plus(py::cast<Polytope>(o)));
        if (py::isinstance<Interval>(o)) return py::cast(A.plus(py::cast<Interval>(o)));
        if (const std::optional<Tensor> v = as_tensor(o)) return py::cast(A.plus(*v));
        return not_implemented();
    };
    // the sum is commutative; the difference only shifts
    method(P, "__add__", polytope_sum);
    method(P, "__radd__", polytope_sum);
    method(P, "__sub__", [](const Polytope &A, const py::object &o) -> py::object {
        if (const std::optional<Tensor> v = as_tensor(o)) return py::cast(A.minus(*v));
        return not_implemented();
    });
    // a matrix maps the set from the left
    method(P, "__rmul__", [](const Polytope &A, const py::object &o) -> py::object {
        if (const std::optional<Tensor> M = as_tensor(o)) return py::cast(*M * A);
        return not_implemented();
    });
    method(P, "__rmatmul__", [](const Polytope &A, const py::object &o) -> py::object {
        if (const std::optional<Tensor> M = as_tensor(o)) return py::cast(*M * A);
        return not_implemented();
    });

    // ConZonotope: a point shifts it, a matrix or number maps it
    const auto con_zonotope_sum = [](const ConZonotope &A, const py::object &o) -> py::object {
        if (py::isinstance<ContSet>(o)) return object_of(plus(A, set_of(o)));
        if (const std::optional<Tensor> v = as_tensor(o)) return py::cast(A + *v);
        return not_implemented();
    };
    method(CZ, "__add__", con_zonotope_sum);
    method(CZ, "__radd__", con_zonotope_sum);
    method(CZ, "__sub__", [](const ConZonotope &A, const py::object &o) -> py::object {
        if (const std::optional<Tensor> v = as_tensor(o)) return py::cast(A - *v);
        return not_implemented();
    });
    const auto con_zonotope_product = [](const ConZonotope &A, const py::object &o) -> py::object {
        if (is_number(o)) {
            const double s = py::cast<double>(o);
            return py::cast(ConZonotope(A.c() * s, A.generators() * s, A.A(), A.b()));
        }
        if (const std::optional<Tensor> M = as_tensor(o)) return py::cast(*M * A);
        return not_implemented();
    };
    method(CZ, "__rmul__", con_zonotope_product);
    // the right operators of the constrained zonotope
    method(CZ, "__rmatmul__", con_zonotope_product);

    // ZonoBundle: a point shifts it, a set adds to it, a matrix or number maps it
    const py::object ZB = m.attr("ZonoBundle");
    ZB.attr("__array_ufunc__") = py::none();
    const auto bundle_sum = [](const ZonoBundle &A, const py::object &o) -> py::object {
        if (py::isinstance<ContSet>(o)) return object_of(plus(A, set_of(o)));
        if (const std::optional<Tensor> v = as_tensor(o)) return py::cast(A.plus(*v));
        return not_implemented();
    };
    method(ZB, "__add__", bundle_sum);
    method(ZB, "__radd__", bundle_sum);
    const auto bundle_product = [](const ZonoBundle &A, const py::object &o) -> py::object {
        if (is_number(o))
            return py::cast(A.linear_map(A.zonotopes().front().center().eye_like(A.dim()) *
                                         py::cast<double>(o)));
        if (const std::optional<Tensor> M = as_tensor(o)) return py::cast(A.linear_map(*M));
        return not_implemented();
    };
    method(ZB, "__rmul__", bundle_product);
    method(ZB, "__rmatmul__", bundle_product);

    // Zonotope
    const auto zonotope_sum = [](const Zonotope &A, const py::object &o) -> py::object {
        if (py::isinstance<ContSet>(o)) return object_of(plus(A, set_of(o)));
        if (const std::optional<Tensor> v = as_tensor(o)) return py::cast(A + *v);
        return not_implemented();
    };
    method(Z, "__add__", zonotope_sum);
    method(Z, "__radd__", zonotope_sum);
    method(Z, "__sub__", [](const Zonotope &A, const py::object &o) -> py::object {
        if (const std::optional<Tensor> v = as_tensor(o)) return py::cast(A - *v);
        return not_implemented();
    });
    method(Z, "__neg__", [](const Zonotope &A) { return -A; });
    method(Z, "__mul__", [](const Zonotope &A, const py::object &o) -> py::object {
        if (is_number(o)) return py::cast(A * py::cast<double>(o));
        return not_implemented();
    });
    const auto zonotope_product = [](const Zonotope &A, const py::object &o) -> py::object {
        if (is_number(o)) return py::cast(py::cast<double>(o) * A);
        if (const std::optional<Tensor> M = as_tensor(o)) return py::cast(*M * A);
        // a number or a matrix scales the zonotope
        return not_implemented();
    };
    method(Z, "__rmul__", zonotope_product);
    method(Z, "__rmatmul__", zonotope_product);

    // Interval
    const auto interval_sum = [](const Interval &A, const py::object &o) -> py::object {
        if (py::isinstance<ContSet>(o)) return object_of(plus(A, set_of(o)));
        if (const std::optional<Tensor> v = as_tensor(o)) return py::cast(A + *v);
        return not_implemented();
    };
    method(I, "__add__", interval_sum);
    method(I, "__radd__", interval_sum);
    method(I, "__sub__", [](const Interval &A, const py::object &o) -> py::object {
        if (py::isinstance<Interval>(o)) return py::cast(A - py::cast<Interval>(o));
        if (const std::optional<Tensor> v = as_tensor(o)) return py::cast(A - *v);
        return not_implemented();
    });
    method(I, "__rsub__", [](const Interval &A, const py::object &o) -> py::object {
        if (const std::optional<Tensor> v = as_tensor(o)) return py::cast(*v - A);
        return not_implemented();
    });
    method(I, "__truediv__", [](const Interval &A, const py::object &o) -> py::object {
        if (is_number(o)) return py::cast(A / py::cast<double>(o));
        return not_implemented();
    });
    // a number or a matrix divided by an interval
    method(I, "__rtruediv__", [](const Interval &A, const py::object &o) -> py::object {
        if (is_number(o) || as_tensor(o))
            return py::cast(Interval::rdivide(operand(A, o, "the numerator"), A));
        return not_implemented();
    });
    method(I, "__pow__", [](const Interval &A, const py::object &o) -> py::object {
        if (py::isinstance<Interval>(o)) return py::cast(A.power(py::cast<Interval>(o)));
        if (is_number(o)) return py::cast(A.power(py::cast<double>(o)));
        if (const std::optional<Tensor> e = as_tensor(o)) return py::cast(A.power(*e));
        return not_implemented();
    });
    method(I, "__rpow__", [](const Interval &A, const py::object &o) -> py::object {
        if (is_number(o) || as_tensor(o))
            return py::cast(Interval::power(operand(A, o, "the base"), A));
        return not_implemented();
    });
    method(I, "__le__", [](const Interval &A, const py::object &o) -> py::object {
        if (py::isinstance<Interval>(o)) return py::cast(A <= py::cast<Interval>(o));
        return not_implemented();
    });
    // comparison with another interval
    method(I, "__lt__", [](const Interval &A, const py::object &o) -> py::object {
        if (py::isinstance<Interval>(o)) return py::cast(A < py::cast<Interval>(o));
        return not_implemented();
    });
    method(I, "__neg__", [](const Interval &A) { return -A; });
    method(I, "__pos__", [](const Interval &A) { return +A; });
    const auto interval_map = [](const Interval &A, const py::object &o) -> py::object {
        if (is_number(o)) return py::cast(A * py::cast<double>(o));
        if (py::isinstance<Zonotope>(o)) return py::cast(A * py::cast<Zonotope>(o));
        return not_implemented();
    };
    method(I, "__mul__", interval_map);
    method(I, "__matmul__", interval_map);
    const auto interval_product = [](const Interval &A, const py::object &o) -> py::object {
        if (is_number(o)) return py::cast(py::cast<double>(o) * A);
        if (const std::optional<Tensor> M = as_tensor(o)) return py::cast(*M * A);
        return not_implemented();
    };
    method(I, "__rmul__", interval_product);
    method(I, "__rmatmul__", interval_product);

    // Capsule: a matrix maps it
    const auto capsule_product = [](const Capsule &A, const py::object &o) -> py::object {
        if (const std::optional<Tensor> M = as_tensor(o)) return py::cast(*M * A);
        return not_implemented();
    };
    // Capsule and Ellipsoid: a vector or a set adds
    const auto capsule_sum = [](const Capsule &A, const py::object &o) -> py::object {
        if (py::isinstance<ContSet>(o)) return object_of(plus(A, set_of(o)));
        if (const std::optional<Tensor> v = as_tensor(o)) return py::cast(A.plus(*v));
        return not_implemented();
    };
    method(m.attr("Capsule"), "__add__", capsule_sum);
    method(m.attr("Capsule"), "__radd__", capsule_sum);
    const auto ellipsoid_sum = [](const Ellipsoid &A, const py::object &o) -> py::object {
        if (py::isinstance<ContSet>(o)) return object_of(plus(A, set_of(o)));
        if (const std::optional<Tensor> v = as_tensor(o)) return py::cast(A.plus(*v));
        return not_implemented();
    };
    method(m.attr("Ellipsoid"), "__add__", ellipsoid_sum);
    method(m.attr("Ellipsoid"), "__radd__", ellipsoid_sum);
    // a matrix maps an ellipsoid
    const auto ellipsoid_product = [](const Ellipsoid &A, const py::object &o) -> py::object {
        if (const std::optional<Tensor> M = as_tensor(o)) return py::cast(A.linear_map(*M));
        return not_implemented();
    };
    method(m.attr("Ellipsoid"), "__rmul__", ellipsoid_product);
    method(m.attr("Ellipsoid"), "__rmatmul__", ellipsoid_product);
    method(m.attr("Capsule"), "__rmul__", capsule_product);
    method(m.attr("Capsule"), "__rmatmul__", capsule_product);
    // EmptySet
    const auto empty_set_product = [](const EmptySet &A, const py::object &o) -> py::object {
        if (is_number(o)) return py::cast(py::cast<double>(o) * A);
        if (py::isinstance<Interval>(o)) return py::cast(py::cast<Interval>(o) * A);
        if (const std::optional<Tensor> M = as_tensor(o)) return py::cast(*M * A);
        return not_implemented();
    };
    method(E, "__rmul__", empty_set_product);
    method(E, "__rmatmul__", empty_set_product);
    // Fullspace: a point shifts it, a number or matrix maps it
    const auto fullspace_sum = [](const Fullspace &A, const py::object &o) -> py::object {
        if (py::isinstance<ContSet>(o)) return object_of(plus(A, set_of(o)));
        if (as_tensor(o)) return object_of(A.plus(operand_set(o).get()));
        return not_implemented();
    };
    method(F, "__add__", fullspace_sum);
    method(F, "__radd__", fullspace_sum);
    const auto fullspace_product = [](const Fullspace &A, const py::object &o) -> py::object {
        if (is_number(o))
            return object_of(A.linear_map(Tensor::from_data({py::cast<double>(o)}, {1, 1})));
        if (const std::optional<Tensor> M = as_tensor(o)) return object_of(*M * A);
        return not_implemented();
    };
    method(F, "__rmul__", fullspace_product);
    method(F, "__rmatmul__", fullspace_product);
}

// Dynamics ----------------------------------------------------------------------------------------

/// The time points of a simulation as one array, the time first: (steps + 1, ..., n, N).
py::object stack_time(const std::vector<Tensor> &points) {
    if (points.empty()) CORA_FAIL(Error::report_to_dev, "cora: a simulation without time points");
#ifdef CORACPP_PYTHON_TORCH
    if (is_torch(points[0])) {
        std::vector<torch::Tensor> all;
        for (const Tensor &x : points) all.push_back(to_torch(x));
        return py::cast(torch::stack(torch::broadcast_tensors(all), 0));
    }
#endif
    py::list all;
    for (const Tensor &x : points) all.append(matrix(x));
    return py::module_::import("numpy").attr("stack")(all);
}

/// The algorithm name kwargs give (popped from it), else `fallback`.
std::string pop_alg(py::dict &kwargs, const std::string &fallback) {
    if (!kwargs.contains("alg")) return fallback;
    const std::string name = py::cast<std::string>(kwargs["alg"]);
    PyDict_DelItemString(kwargs.ptr(), "alg");
    return name;
}

/// The algorithms registered from Python, {(system, operation, name): function}.
py::dict registered_algorithms() { return py::module_::import("cora.extend").attr("ALGORITHMS"); }

/// The names of the algorithms registered from Python for (system, operation).
std::vector<std::string> registered_names(const char *system, const char *operation) {
    std::vector<std::string> names;
    for (const auto &item : registered_algorithms()) {
        const py::tuple key = py::reinterpret_borrow<py::tuple>(item.first);
        if (py::cast<std::string>(key[0]) == system && py::cast<std::string>(key[1]) == operation)
            names.push_back(py::cast<std::string>(key[2]));
    }
    return names;
}

/// The keyword `key` of an operation, which must be given.
py::object required(const py::kwargs &kwargs, const char *key, const char *operation) {
    if (!kwargs.contains(key))
        CORA_FAIL(Error::wrong_value,
                  std::string(operation) + ": the keyword '" + key + "' is missing");
    return kwargs[key];
}

/// An operation run from keyword arguments: the problem P and an algorithm of the variant V,
/// built in or registered from Python; `extra` names keywords the caller takes itself. One
/// positional argument is the first field of P (the initial set or state).
template <class P, class V, class Run>
py::object run_with_kwargs(const py::object &self, const char *operation, const py::args &args,
                           const py::kwargs &kwargs, const std::vector<std::string> &extra, Run run,
                           const std::function<void(const P &)> &validate = {}) {
    const std::string system_name = py::str(py::type::of(self).attr("__name__"));
    const char *system = system_name.c_str();
    py::dict rest; // a copy: the keys used are removed
    for (const auto &item : kwargs) rest[item.first] = item.second;
    const char *first = std::get<0>(P::fields()).name;
    if (args.size() > 1 || (args.size() == 1 && kwargs.contains(first)))
        CORA_FAIL(Error::wrong_value, std::string(operation) + ": give " + first +
                                          " as the one positional argument, the rest by keyword");
    if (args.size() == 1) rest[first] = args[0];
    for (const std::string &k : extra)
        if (rest.contains(k.c_str())) PyDict_DelItemString(rest.ptr(), k.c_str());
    const std::vector<std::string> builtin = algorithm_names<V>();
    const std::string name = pop_alg(rest, builtin.front());
    const P problem = fill(P{}, rest);
    const std::string where = operation;
    if (std::optional<V> alg = algorithm<V>(name, rest)) {
        // the problem, this algorithm's options and the caller's keywords are known
        std::vector<std::string> known = field_names<P>();
        std::visit(
            [&](const auto &a) {
                for (const std::string &f : field_names<std::decay_t<decltype(a)>>())
                    known.push_back(f);
            },
            *alg);
        for (const std::string &k : extra) known.push_back(k);
        known.emplace_back("alg");
        const std::vector<std::string> others =
            algorithm_fields<V>(std::make_index_sequence<std::variant_size_v<V>>());
        refuse_leftovers(rest, known, where, others, name);
        return run(problem, *alg);
    }
    // an algorithm of Python: the same problem checks, then the function with the other keywords
    const py::dict algorithms = registered_algorithms();
    const py::tuple key = py::make_tuple(system, operation, name);
    if (!algorithms.contains(key) || !validate) {
        std::vector<std::string> names = builtin;
        for (const std::string &r : registered_names(system, operation)) names.push_back(r);
        CORA_FAIL(Error::wrong_value, unknown_algorithm(name, names, where));
    }
    validate(problem);
    return algorithms[key](self, py::cast(problem), **rest);
}

void bind_problems(py::module_ &m) {
    // the reachable sets, with the times of an adaptive step size
    py::class_<ReachSet>(m, "ReachSet", "The reachable sets: lists of zonotopes.")
        .def(py::init([](std::vector<Zonotope> time_interval, std::vector<Zonotope> time_point,
                         double time_step, std::vector<double> time) {
                 // the reachable sets and their step
                 return ReachSet{std::move(time_interval), std::move(time_point), time_step,
                                 std::move(time)};
             }),
             py::arg("time_interval") = std::vector<Zonotope>(),
             py::arg("time_point") = std::vector<Zonotope>(), py::arg("time_step") = 0.0,
             py::arg("time") = std::vector<double>())
        .def_readonly("time_step", &ReachSet::time_step, "the step between the sets; 0 if unknown")
        .def_readonly("time", &ReachSet::time,
                      "the step times t_0, ..., t_K of an adaptive step size; empty otherwise")
        // the sets over the steps and at their ends
        .def_readonly("time_interval", &ReachSet::time_interval,
                      "time_interval[k] over [k*time_step, (k+1)*time_step]")
        .def_readonly("time_point", &ReachSet::time_point, "time_point[k] at k*time_step");

    py::class_<ReachProblem>(m, "ReachProblem", "What reach and verify are asked (CORA's params).")
        .def_readonly("initial_set", &ReachProblem::initial_set, "The initial set X0")
        .def_readonly("input_set", &ReachProblem::input_set, "The input set U, or None")
        .def_readonly("t_final", &ReachProblem::t_final, "The final time");

    py::class_<Falsification>(m, "Falsification", "A trajectory that violates a specification.")
        .def_property_readonly(
            "x0", [](const Falsification &f) { return vector(f.x0); }, "the state at time 0")
        .def_readonly("time", &Falsification::time, "the time it is violated");

    py::class_<VerifyResult>(m, "VerifyResult", "The outcome of verify.")
        .def_readonly("verified", &VerifyResult::verified,
                      "True if the specification holds on the whole horizon")
        .def_readonly("computation_time", &VerifyResult::computation_time, "seconds spent")
        // the fields of the verification result
        .def_readonly("iterations", &VerifyResult::iterations, "passes of the adaptive loop")
        .def_readonly("time_step", &VerifyResult::time_step, "step size of the last pass")
        .def_readonly("num_steps", &VerifyResult::num_steps, "number of steps of the last pass")
        .def_readonly("falsification", &VerifyResult::falsification, "a Falsification, or None");
}

/// The algorithm of the variant named `name`, filled from kwargs; refuses what is left over.
template <class Variant>
Variant step_algorithm(const std::string &name, py::dict kwargs, const std::string &where) {
    const std::optional<Variant> a = algorithm<Variant>(name, kwargs);
    if (!a)
        CORA_FAIL(Error::wrong_value, unknown_algorithm(name, algorithm_names<Variant>(), where));
    std::vector<std::string> known;
    std::visit([&](const auto &alg) { known = field_names<std::decay_t<decltype(alg)>>(); }, *a);
    const std::vector<std::string> others =
        algorithm_fields<Variant>(std::make_index_sequence<std::variant_size_v<Variant>>());
    refuse_leftovers(kwargs, known, where, others, name);
    return *a;
}

/// The linear reach algorithm named `name`, filled from kwargs.
alg::LinearReach linear_reach_algorithm(const std::string &name, py::dict kwargs,
                                        const std::string &where) {
    return step_algorithm<alg::LinearReach>(name, std::move(kwargs), where);
}

/// The result structs of the steps of a linear system, with their sets as properties.
py::class_<CanonicalForm> bind_linear_steps(py::module_ &m) {
    // its system is added once LinearSys is bound
    py::class_<CanonicalForm> canonical(m, "CanonicalForm",
                                        "A system in canonical form with its input.");
    canonical.def_readonly("U", &CanonicalForm::U, "The centered input set in the state space")
        .def_property_readonly(
            "u", [](const CanonicalForm &c) { return matrix(c.u); },
            "The constant input per step (n, steps)");
    py::class_<HomogeneousSolution>(m, "HomogeneousSolution", "The homogeneous solution of a step.")
        .def_readonly("time_point", &HomogeneousSolution::time_point, "The set at the end")
        .def_readonly("time_interval", &HomogeneousSolution::time_interval, "The set over the step")
        .def_readonly("state_curvature", &HomogeneousSolution::state_curvature, "F X");
    // the solutions of a step
    py::class_<ConstantSolution>(m, "ConstantSolution", "The solution of a constant input.")
        .def_readonly("time_point", &ConstantSolution::time_point, "The set at the end")
        .def_readonly("input_curvature", &ConstantSolution::input_curvature, "G u")
        .def_readonly("time_interval", &ConstantSolution::time_interval, "The set over the step");
    // a step with a constant input
    py::class_<AffineSolution>(m, "AffineSolution", "The solution of a step with a constant input.")
        .def_readonly("time_point", &AffineSolution::time_point, "The set at the end")
        .def_readonly("constant", &AffineSolution::constant, "The part of the constant input")
        .def_readonly("time_interval", &AffineSolution::time_interval, "The set over the step")
        .def_readonly("state_curvature", &AffineSolution::state_curvature, "F X")
        .def_readonly("input_curvature", &AffineSolution::input_curvature, "G u");
    // one step of reach, and what it hands on
    py::class_<OneStep>(m, "OneStep", "One step of reach with the sets it is made of.")
        .def_readonly("time_point", &OneStep::time_point, "The reachable set at the end")
        .def_readonly("time_interval", &OneStep::time_interval, "The reachable set over the step")
        // the parts the sets of the step are made of
        .def_readonly("homogeneous_time_point", &OneStep::homogeneous_time_point,
                      "The homogeneous solution at the end plus the constant input")
        .def_readonly("homogeneous_time_interval", &OneStep::homogeneous_time_interval,
                      "The enclosure of the step without the time-varying input")
        // the input solutions and the curvatures
        .def_readonly("time_varying", &OneStep::time_varying, "The time-varying input solution")
        .def_readonly("constant", &OneStep::constant, "The constant input solution")
        .def_readonly("state_curvature", &OneStep::state_curvature, "F X")
        .def_readonly("input_curvature", &OneStep::input_curvature, "G u");
    // the error of a linearized trajectory
    py::class_<TrajectoryError>(m, "TrajectoryError", "The error of a linearized trajectory.")
        .def_readonly("error", &TrajectoryError::error, "The set that encloses the error")
        .def_readonly("c_tilde", &TrajectoryError::c_tilde,
                      "The box of the coefficients A_tilde' n")
        .def_readonly("d_tilde", &TrajectoryError::d_tilde, "The box of the offset -n' u_tilde");
    // what a step hands on to the next
    py::class_<ReachState>(m, "ReachState", "What a step of init_reach and post hands on.")
        .def_readonly("homogeneous", &ReachState::homogeneous, "The homogeneous set over the step")
        .def_readonly("homogeneous_time_point", &ReachState::homogeneous_time_point,
                      "The homogeneous set at the end")
        .def_readonly("aux", &ReachState::aux, "The input solution of the first step, mapped on")
        .def_readonly("par", &ReachState::par, "The sum of the input solutions so far")
        .def_readonly("trans", &ReachState::trans, "The constant part of the input");
    // one step of init_reach or post
    py::class_<ReachStep>(m, "ReachStep", "One step of init_reach or post.")
        .def_readonly("time_interval", &ReachStep::time_interval, "The reachable set over the step")
        .def_readonly("time_point", &ReachStep::time_point, "The reachable set at the end")
        .def_readonly("state", &ReachStep::state, "The state for the next step");
    return canonical;
}

void bind_linear_sys(py::module_ &m) {
    // the results of the steps first, then the system
    py::class_<CanonicalForm> canonical = bind_linear_steps(m);
    py::class_<LinearSys>(m, "LinearSys", "The linear system x' = A x + B u, y = C x.")
        .def(py::init([](const Array &A, const py::object &B, const py::object &C,
                         bool custom_backward) {
                 // custom_backward: hand-written backward pass of the matrix exponential (torch)
                 const auto t = [&](const py::object &o, const char *what) {
#ifdef CORACPP_PYTHON_TORCH
                     if (custom_backward) return from_torch(py::cast<torch::Tensor>(o), true);
#endif
                     if (custom_backward)
                         CORA_FAIL(Error::install, "custom_backward needs the torch build");
                     return tensor_of(o, what);
                 };
                 if (B.is_none()) return LinearSys(t(A, "A"));
                 // the output matrix is optional
                 return LinearSys(t(A, "A"), t(B, "B"),
                                  C.is_none() ? std::nullopt : std::optional<Tensor>(t(C, "C")));
             }),
             py::arg("A"), py::arg("B") = py::none(), py::arg("C") = py::none(),
             py::arg("custom_backward") = false,
             "x' = A x (+ B u), y = C x: A (..., n, n), B (n, m), C (p, n); without C the output "
             "is x")
        .def_property_readonly(
            "A", [](const LinearSys &sys) { return matrix(sys.A()); },
            "The system matrix A (..., n, n)")
        .def_property_readonly(
            "B",
            [](const LinearSys &sys) -> std::optional<Array> {
                if (!sys.B()) return std::nullopt;
                return matrix(*sys.B());
            },
            "The input matrix B (n, m), or None")
        .def_property_readonly(
            "C",
            [](const LinearSys &sys) -> std::optional<Array> {
                // the output matrix is optional
                if (!sys.C()) return std::nullopt;
                return matrix(*sys.C());
            },
            "The output matrix C (p, n), or None (then the output is x)")
        .def_property_readonly("dim", &LinearSys::dim, "The number of states n")
        .def(
            "reach",
            [](const py::object &self, const py::args &args, const py::kwargs &kwargs) {
                const LinearSys &sys = py::cast<const LinearSys &>(self);
                return run_with_kwargs<ReachProblem, alg::LinearReach>(
                    self, "reach", args, kwargs, {},
                    [&](const ReachProblem &p, const alg::LinearReach &a) {
                        return py::cast(sys.reach(p, a));
                    },
                    [&](const ReachProblem &p) { sys.validate(p); });
            },
            "reach(initial_set=X0, t_final=T, alg='standard', time_step=dt, taylor_terms=10, "
            "zonotope_order=50, input_set=U): the reachable sets; alg 'standard' or "
            "'wrapping_free', or one registered with cora.register_algorithm")
        // output sets of the reachable sets
        .def("output_set", &LinearSys::output_set, py::arg("R"),
             "The output sets y = C x of the reachable sets R")
        .def(
            "simulate",
            [](const py::object &self, const py::args &args, const py::kwargs &kwargs) {
                const LinearSys &sys = py::cast<const LinearSys &>(self);
                return run_with_kwargs<SimulationProblem, alg::LinearSimulate>(
                    self, "simulate", args, kwargs, {},
                    [&](const SimulationProblem &p, const alg::LinearSimulate &a) {
                        return stack_time(sys.simulate(p, a));
                    });
            },
            "simulate(initial_state=x0 (..., n, N), t_final=T, time_step=dt, input=u (m, N) or "
            "(m,)): the states (steps + 1, ..., n, N), time first; exact")
        // random trajectories from points of the sets
        .def(
            "simulate_random",
            [](const py::object &self, const py::args &args, const py::kwargs &kwargs) {
                const LinearSys &sys = py::cast<const LinearSys &>(self);
                const int64_t points =
                    py::cast<int64_t>(required(kwargs, "points", "simulate_random"));
                // random simulation needs the points and the generator
                cora::Rng &rng = py::cast<cora::Rng &>(required(kwargs, "rng", "simulate_random"));
                return run_with_kwargs<RandomSimulationProblem, alg::LinearSimulate>(
                    self, "simulate_random", args, kwargs, {"points", "rng"},
                    [&](const RandomSimulationProblem &p, const alg::LinearSimulate &a) {
                        return stack_time(sys.simulate_random(p, a, points, rng));
                    });
            },
            "simulate_random(initial_set=X0, t_final=T, time_step=dt, points=N, rng=rng, "
            "input_set=U): simulate from N random points of X0 (and constant inputs of U)")
        // the matrices of one step
        .def("correction_matrix_state", &LinearSys::correction_matrix_state, py::arg("time_step"),
             py::arg("taylor_terms"), "The interval matrix F of the curvature enlargement")
        .def("correction_matrix_input", &LinearSys::correction_matrix_input, py::arg("time_step"),
             py::arg("taylor_terms"), "The interval matrix G of the curvature of the input")
        .def("expm_remainder", &LinearSys::expm_remainder, py::arg("time_step"),
             py::arg("taylor_terms"), "The error box of the truncated series of e^{|A| dt}")
        // canonical form, with the constant input per step
        .def(
            "canonical_form",
            [](const LinearSys &sys, const Zonotope &U, const py::object &u_vec) {
                return sys.canonical_form(
                    U, u_vec.is_none() ? std::nullopt
                                       : std::optional<Tensor>(tensor_of(u_vec, "u_vec")));
            },
            py::arg("U"), py::arg("u_vec") = py::none(),
            "The system with B = I, the centered input set and the constant input per step")
        // canonical form
        .def("is_canonical_form", &LinearSys::is_canonical_form,
             "Whether B and C are identity matrices")
        // the solutions of one step
        .def("homogeneous_solution", &LinearSys::homogeneous_solution, py::arg("X"),
             py::arg("time_step"), py::arg("taylor_terms"),
             "The solution of x' = A x over one step: the set at its end, over it and F X")
        // the input solutions: the terms apart, and for a constant input
        .def("particular_solution_time_varying", &LinearSys::particular_solution_time_varying,
             py::arg("U"), py::arg("time_step"), py::arg("taylor_terms"),
             "What the input u(t) in U adds over one step, the terms of A kept apart")
        .def("particular_solution_constant", &LinearSys::particular_solution_constant, py::arg("U"),
             py::arg("time_step"), py::arg("taylor_terms"),
             "What a constant input in U adds over one step, with its curvature")
        // the solution with a constant input u
        .def(
            "affine_solution",
            [](const LinearSys &sys, const Zonotope &X, const py::object &u, double time_step,
               int taylor_terms) {
                return sys.affine_solution(X, tensor_of(u, "u"), time_step, taylor_terms);
            },
            py::arg("X"), py::arg("u"), py::arg("time_step"), py::arg("taylor_terms"),
            "The solution of one step of x' = A x + u for the constant input u (n,)")
        // a whole step of reach
        .def(
            "one_step",
            [](const LinearSys &sys, const Zonotope &X, const Zonotope &U, const py::object &u,
               double time_step, int taylor_terms) {
                return sys.one_step(X, U, tensor_of(u, "u"), time_step, taylor_terms);
            },
            py::arg("X"), py::arg("U"), py::arg("u"), py::arg("time_step"), py::arg("taylor_terms"),
            "One step of reach from X with the input set U and the constant input u (n,)")
        // the first step and the steps after it
        .def(
            "init_reach",
            [](const LinearSys &sys, const Zonotope &R_init, const std::optional<Zonotope> &U,
               const py::object &u_trans, const std::string &alg, const py::kwargs &kwargs) {
                py::dict options;
                for (const auto &item : kwargs) options[item.first] = item.second;
                return sys.init_reach(R_init, U,
                                      u_trans.is_none()
                                          ? std::nullopt
                                          : std::optional<Tensor>(tensor_of(u_trans, "u_trans")),
                                      linear_reach_algorithm(alg, options, "init_reach"));
            },
            py::arg("R_init"), py::arg("U") = py::none(), py::arg("u_trans") = py::none(),
            py::arg("alg") = "standard",
            "The first step of the algorithm 'standard' or 'wrapping_free' (options as keywords)")
        // the steps after the first
        .def(
            "post",
            [](const LinearSys &sys, const ReachState &state, const std::string &alg,
               const py::kwargs &kwargs) {
                py::dict options;
                for (const auto &item : kwargs) options[item.first] = item.second;
                return sys.post(state, linear_reach_algorithm(alg, options, "post"));
            },
            py::arg("state"), py::arg("alg") = "standard",
            "The step after state, with the algorithm and options of init_reach")
        // the adaptive error of the input solution
        .def(
            "error_solution_adaptive",
            [](const LinearSys &sys, const Zonotope &Vdyn, double time_step, double zeta_tabs,
               const std::optional<Zonotope> &Vstat, const std::optional<int> &taylor_terms) {
                return sys.error_solution_adaptive(Vdyn, Vstat, time_step, zeta_tabs, taylor_terms);
            },
            py::arg("Vdyn"), py::arg("time_step"), py::arg("zeta_tabs"),
            py::arg("Vstat") = py::none(), py::arg("taylor_terms") = py::none(),
            "The error of the input solution over a step, cut where the series gains little")
        // the error of the linearization
        .def(
            "trajectory_error",
            [](const LinearSys &sys, const py::object &f, const py::object &u, const Zonotope &R0,
               double t_min, double t_max, double t_c, int order, const py::object &n) {
                // one flow value, or the values at the start and the end
                std::vector<Tensor> flows;
                if (py::isinstance<py::list>(f) || py::isinstance<py::tuple>(f))
                    for (const py::handle &x : f) flows.push_back(tensor_of(x, "f"));
                else
                    flows.push_back(tensor_of(f, "f"));
                return sys.trajectory_error(flows, tensor_of(u, "u"), R0, t_min, t_max, t_c, order,
                                            tensor_of(n, "n"));
            },
            py::arg("f"), py::arg("u"), py::arg("R0"), py::arg("t_min"), py::arg("t_max"),
            py::arg("t_c"), py::arg("order"), py::arg("n"),
            "The error of the trajectories against the linearization around t_c, as a set and two "
            "boxes")
        // equality, text and random systems
        .def("isequal", &LinearSys::isequal, py::arg("other"), py::arg("tol") = py::none(),
             "Whether the matrices agree up to tol (default: the repository tolerance)")
        // equal systems have equal matrices
        .def("__eq__",
             [](const LinearSys &a, const py::object &b) -> py::object {
                 if (!py::isinstance<LinearSys>(b)) return not_implemented();
                 return py::cast(a == b.cast<const LinearSys &>());
             })
        .def("__ne__",
             [](const LinearSys &a, const py::object &b) -> py::object {
                 if (!py::isinstance<LinearSys>(b)) return not_implemented();
                 return py::cast(a != b.cast<const LinearSys &>());
             })
        .def("__repr__", &LinearSys::display)
        .def("__str__", &LinearSys::display)
        // a random stable system
        .def_static("generate_random", &LinearSys::generate_random, py::arg("rng"),
                    py::arg("state_dimension") = py::none(),
                    py::arg("input_dimension") = py::none(),
                    py::arg("output_dimension") = py::none(), py::arg("real_interval") = py::none(),
                    py::arg("imaginary_interval") = py::none(),
                    "A random system with stable eigenvalues in a box of the complex plane")
        // the system as a nonlinear one
        .def("nonlinear_sys", &LinearSys::nonlinear_sys,
             "The system x' = A x as a NonlinearSys (without inputs)")
        .def("validate", &LinearSys::validate, py::arg("problem"),
             "The checks of a reach problem against the system")
        .def(
            "verify",
            [](const py::object &self, const py::args &args, const py::kwargs &kwargs) {
                // verify needs the specifications
                const LinearSys &sys = py::cast<const LinearSys &>(self);
                const std::vector<Specification> specs =
                    py::cast<std::vector<Specification>>(required(kwargs, "specs", "verify"));
                return run_with_kwargs<ReachProblem, alg::LinearVerify>(
                    self, "verify", args, kwargs, {"specs"},
                    [&](const ReachProblem &p, const alg::LinearVerify &a) {
                        return py::cast(sys.verify(p, specs, a));
                    });
            },
            "verify(initial_set=R0, input_set=U, t_final=T, specs=[...], "
            "alg='reach_avoid_support_function', 'reach_avoid_zonotope' or 'reach_avoid_krylov'): "
            "whether every "
            "specification holds for the outputs up to t_final");
    // a canonical form holds a system, bound just now
    canonical.def_readonly("sys", &CanonicalForm::sys, "The system with B = I");
}

// LevelSet and Taylm ------------------------------------------------------------------------------

/// The comparison of a constraint from its CORA string.
CompOp comp_op_of(const std::string &op) {
    if (op == "<=") return CompOp::le;
    if (op == "<") return CompOp::lt;
    if (op == "==") return CompOp::eq;
    CORA_FAIL(Error::wrong_value, "LevelSet: a comparison is '<=', '<' or '==', got '" + op + "'");
}

/// The CORA string of a comparison.
const char *comp_op_name(CompOp op) {
    return op == CompOp::le ? "<=" : op == CompOp::lt ? "<" : "==";
}

/// A point or a matrix of points from Python: a vector is one point.
Tensor points_of(const py::object &o) {
    const py::array a = py::array_t<double, py::array::forcecast>::ensure(o);
    return a && a.ndim() == 1 ? vector_of(o, "the point") : tensor_of(o, "the points");
}

/// An invariant or a guard from Python: a level set or a polytope.
Region region_of(const py::object &o) {
    if (py::isinstance<LevelSet>(o)) return py::cast<LevelSet>(o);
    return py::cast<Polytope>(o);
}

void bind_level_set(py::module_ &m) {
    // a level set is built from expressions and strings, and queried with arrays or boxes
    py::class_<LevelSet, ContSet>(m, "LevelSet",
                                  "The level set {x | f_i(x) ~ 0}: every constraint f_i (an Expr "
                                  "of the states) holds with its comparison '<=', '<' or '=='.")
        .def(py::init(
                 [](const std::vector<Expr> &eq, int64_t n, const std::vector<std::string> &ops) {
                     std::vector<CompOp> o;
                     o.reserve(ops.size());
                     for (const std::string &s : ops) o.push_back(comp_op_of(s));
                     return LevelSet(eq, n, std::move(o));
                 }),
             py::arg("eq"), py::arg("n"), py::arg("ops"),
             "Constraints eq[i] ops[i] 0 on n states; ops are '<=', '<' or '=='")
        .def(py::init([](const Expr &eq, int64_t n, const std::string &op) {
                 return LevelSet(eq, n, comp_op_of(op));
             }),
             py::arg("eq"), py::arg("n"), py::arg("op"), "One constraint eq op 0 on n states")
        // the two constants, the constraints and the queries
        .def_static("empty", &LevelSet::empty, py::arg("n") = 0, "The empty set of n dimensions")
        .def_static("inf", &LevelSet::inf, py::arg("n") = 0, "The whole space of n dimensions")
        .def_property_readonly("num_equations", &LevelSet::num_equations,
                               "The number of constraints")
        .def_property_readonly(
            "comp_ops",
            [](const LevelSet &L) {
                std::vector<std::string> out;
                for (const CompOp op : L.comp_ops()) out.emplace_back(comp_op_name(op));
                return out;
            },
            "The comparisons of the constraints")
        // the values and the derivatives of the constraints
        .def("gradient", &LevelSet::gradient, py::arg("i"), py::arg("j"),
             "The derivative of constraint i with respect to state j")
        // fun_han takes points or a box
        .def(
            "fun_han",
            [](const LevelSet &L, const py::object &x) -> py::object {
                if (py::isinstance<Interval>(x)) return py::cast(L.fun_han(py::cast<Interval>(x)));
                return matrix(L.fun_han(points_of(x)));
            },
            py::arg("x"), "The constraints at the points x (n, N), or over the box x")
        // contains takes points or a box
        .def(
            "contains",
            [](const LevelSet &L, const py::object &S, Tol tol) {
                if (py::isinstance<Interval>(S)) return L.contains(py::cast<Interval>(S), tol);
                return L.contains(points_of(S), tol);
            },
            py::arg("S"), py::arg("tol") = py::none(),
            "Whether all points (n, N), or the box S, satisfy every constraint")
        // is_intersecting goes through the box of the other set
        .def(
            "is_intersecting",
            [](const LevelSet &L, const py::object &S, Tol tol) {
                return L.is_intersecting(set_of(S).get(), tol);
            },
            py::arg("S"), py::arg("tol") = py::none(),
            "False if the box of S shows that S and the level set are disjoint")
        // the contractor, the complement and the split
        .def("tighten_domain", &LevelSet::tighten_domain, py::arg("I"),
             py::arg("type") = "forwardBackward",
             "The box I cut to the solutions of f = 0, or None if there are none")
        // the intersection with a polynomial zonotope
        .def("intersect", &LevelSet::intersect, py::arg("S"),
             "A polynomial zonotope that holds the intersection with S (one '==' constraint), or "
             "None if the contraction of the box of S shows it empty")
        // the complement of an inequality
        .def("complement", &LevelSet::complement,
             "The complement of a single '<' or '<=' constraint")
        // the equality part and the rest
        .def("split_equality", &LevelSet::split_equality,
             "The equality part and the inequality part, None for a part without constraints");
}

void bind_taylm(py::module_ &m) {
    py::class_<Taylm>(m, "Taylm",
                      "A Taylor model p(a) + [r1, r2] over variables a_i in [-1, 1]: an enclosure "
                      "of the range of an expression.")
        // the factories, one model per coordinate
        .def_static("constant", &Taylm::constant, py::arg("c"), py::arg("vars"),
                    py::arg("max_order") = Taylm::kDefaultMaxOrder,
                    "The constant c in vars variables")
        .def_static("from_interval", &Taylm::from_interval, py::arg("I"),
                    py::arg("max_order") = Taylm::kDefaultMaxOrder,
                    "One model per coordinate of the box I")
        .def_static("from_zonotope", &Taylm::from_zonotope, py::arg("Z"),
                    py::arg("max_order") = Taylm::kDefaultMaxOrder,
                    "One model per coordinate of the zonotope Z")
        .def_static("from_range", &Taylm::from_range, py::arg("lo"), py::arg("hi"), py::arg("vars"),
                    py::arg("max_order") = Taylm::kDefaultMaxOrder,
                    "The model with no dependency that holds [lo, hi]")
        // an expression on models, and the data of a model
        .def_static("evaluate", &Taylm::evaluate, py::arg("f"), py::arg("x"),
                    "The expression f on the models x, one per state")
        .def_property_readonly("num_vars", &Taylm::num_vars, "The number of variables")
        .def_property_readonly("max_order", &Taylm::max_order, "The degree of the polynomial part")
        // the range and the arithmetic
        .def(
            "bounds",
            [](const Taylm &t) {
                const auto b = t.bounds();
                return std::make_pair(b[0], b[1]);
            },
            "The bounds (lo, hi) of the model")
        .def("interval", &Taylm::interval, "The bounds as a box of one dimension")
        .def("plus", py::overload_cast<const Taylm &>(&Taylm::plus, py::const_), py::arg("S"),
             "The sum with a model")
        .def("plus", py::overload_cast<double>(&Taylm::plus, py::const_), py::arg("s"),
             "The sum with a number")
        .def("minus", py::overload_cast<const Taylm &>(&Taylm::minus, py::const_), py::arg("S"),
             "The difference with a model")
        .def("minus", py::overload_cast<double>(&Taylm::minus, py::const_), py::arg("s"),
             "The difference with a number")
        // products, powers and the operators
        .def("times", py::overload_cast<const Taylm &>(&Taylm::times, py::const_), py::arg("S"),
             "The product with a model")
        .def("times", py::overload_cast<double>(&Taylm::times, py::const_), py::arg("s"),
             "The product with a number")
        .def("power", &Taylm::power, py::arg("n"), "The n-th power, n >= 0")
        .def("uminus", &Taylm::uminus, "The negated model")
        .def("__add__", [](const Taylm &a, const Taylm &b) { return a + b; })
        .def("__sub__", [](const Taylm &a, const Taylm &b) { return a - b; })
        .def("__mul__", [](const Taylm &a, const Taylm &b) { return a * b; })
        .def("__neg__", [](const Taylm &a) { return -a; })
        .def("__pow__", &Taylm::power, py::arg("n"));
}

/// The dynamics as symbolic expressions: `x` is a list of Expr, `f(x)` a list of Expr or numbers.
void bind_nonlinear_sys(py::module_ &m) {
    // the results of the steps
    py::class_<Linearized>(m, "Linearized", "The affine system of a step (CORA's linearize).")
        .def_property_readonly(
            "x_star", [](const Linearized &l) { return vector(l.x_star); },
            "The expansion point of the state")
        .def_property_readonly(
            "u_star", [](const Linearized &l) { return vector(l.u_star); },
            "The expansion point of the input")
        .def_property_readonly(
            "f0", [](const Linearized &l) { return vector(l.f0); }, "f at the expansion point")
        // the affine system and its input
        .def_readonly("sys", &Linearized::sys, "x' = A x + u")
        .def_readonly("U", &Linearized::U, "The centered input set in the state space")
        .def_property_readonly(
            "u_const", [](const Linearized &l) { return vector(l.u_const); }, "The constant input");
    // one step of reach
    py::class_<NonlinearStep>(m, "NonlinearStep", "One step of init_reach or post.")
        .def_readonly("time_interval", &NonlinearStep::time_interval, "The set over the step")
        .def_readonly("time_point", &NonlinearStep::time_point, "The set at the end")
        .def_property_readonly(
            "error", [](const NonlinearStep &s) { return vector(s.error); },
            "The linearization error of the step");
    py::class_<Expr>(m, "Expr",
                     "A symbolic expression of the states, differentiated by the system.")
        .def(py::init<double>(), py::arg("value"))
        // the variables, the arithmetic and the derivative
        .def_static("var", &Expr::var, py::arg("i"), "The i-th state")
        .def("__add__", [](const Expr &a, const Expr &b) { return a + b; })
        .def("__radd__", [](const Expr &a, const Expr &b) { return b + a; })
        .def("__sub__", [](const Expr &a, const Expr &b) { return a - b; })
        .def("__rsub__", [](const Expr &a, const Expr &b) { return b - a; })
        .def("__mul__", [](const Expr &a, const Expr &b) { return a * b; })
        .def("__rmul__", [](const Expr &a, const Expr &b) { return b * a; })
        .def("__truediv__", [](const Expr &a, const Expr &b) { return a / b; })
        .def("__rtruediv__", [](const Expr &a, const Expr &b) { return b / a; })
        .def("__neg__", [](const Expr &a) { return -a; })
        .def(
            "__pow__", [](const Expr &a, int n) { return pow(a, n); }, py::arg("n"))
        .def(
            "__pow__", [](const Expr &a, double p) { return pow(a, p); }, py::arg("p"))
        .def("diff", &Expr::diff, py::arg("i"), "The derivative with respect to the i-th state")
        .def("eval", py::overload_cast<const std::vector<double> &>(&Expr::eval, py::const_),
             py::arg("x"), "The value at the point x (a list of numbers)");
    py::implicitly_convertible<double, Expr>();
    // numbers convert to expressions
    py::implicitly_convertible<int, Expr>();
    // the functions of expressions
    m.def(
        "sin", [](const Expr &a) { return sin(a); }, py::arg("e"),
        "The sine of a symbolic expression");
    m.def(
        "cos", [](const Expr &a) { return cos(a); }, py::arg("e"),
        "The cosine of a symbolic expression");
    m.def(
        "exp", [](const Expr &a) { return exp(a); }, py::arg("e"),
        "The exponential of a symbolic expression");
    m.def(
        "tan", [](const Expr &a) { return tan(a); }, py::arg("e"),
        "The tangent of a symbolic expression");
    m.def(
        "sqrt", [](const Expr &a) { return sqrt(a); }, py::arg("e"),
        "The square root of a symbolic expression");

    py::class_<NonlinearSys>(m, "NonlinearSys", "The nonlinear system x' = f(x, u).")
        .def(
            py::init([](const py::function &f, int64_t dim, int64_t inputs) {
                // f runs once, here, on symbolic states (and inputs).
                if (inputs == 0)
                    return NonlinearSys(
                        [&f](const std::vector<Expr> &x) { return f(x).cast<std::vector<Expr>>(); },
                        dim);
                return NonlinearSys(
                    [&f](const std::vector<Expr> &x, const std::vector<Expr> &u) {
                        return f(x, u).cast<std::vector<Expr>>();
                    },
                    dim, inputs);
            }),
            py::arg("f"), py::arg("dim"), py::arg("inputs") = 0,
            "f(x) (f(x, u) with inputs > 0) returns the dim components of the right-hand side")
        .def_property_readonly("dim", &NonlinearSys::dim, "The number of states n")
        .def_property_readonly("input_dim", &NonlinearSys::input_dim, "The number of inputs m")
        // the linearization and the steps of reach
        .def(
            "linearize",
            [](const NonlinearSys &sys, const Zonotope &R, double time_step,
               const std::optional<Zonotope> &U) { return sys.linearize(R, U, time_step); },
            py::arg("R"), py::arg("time_step"), py::arg("U") = py::none(),
            "The affine system of a step from R with the input set U (CORA's linearize)")
        .def(
            "init_reach",
            [](const NonlinearSys &sys, const Zonotope &R0, const std::optional<Zonotope> &U,
               const std::string &alg, const py::kwargs &kwargs) {
                py::dict options;
                for (const auto &item : kwargs) options[item.first] = item.second;
                // the options as keywords, as LinearSys.init_reach takes them
                return sys.init_reach(
                    R0, U, step_algorithm<alg::NonlinearReach>(alg, options, "init_reach"));
            },
            py::arg("R0"), py::arg("U") = py::none(), py::arg("alg") = "linearization",
            "The first step of reach (options as keywords)")
        .def(
            "post",
            [](const NonlinearSys &sys, const NonlinearStep &step, const std::optional<Zonotope> &U,
               const std::string &alg, const py::kwargs &kwargs) {
                py::dict options;
                for (const auto &item : kwargs) options[item.first] = item.second;
                return sys.post(step, U, step_algorithm<alg::NonlinearReach>(alg, options, "post"));
            },
            py::arg("step"), py::arg("U") = py::none(), py::arg("alg") = "linearization",
            "The step after `step`, with the options of init_reach")
        // reach and the simulations, with keyword arguments
        .def(
            "reach",
            [](const py::object &self, const py::args &args, const py::kwargs &kwargs) {
                const NonlinearSys &sys = py::cast<const NonlinearSys &>(self);
                return run_with_kwargs<ReachProblem, alg::NonlinearReach>(
                    self, "reach", args, kwargs, {},
                    [&](const ReachProblem &p, const alg::NonlinearReach &a) {
                        // reach of a nonlinear system with keyword arguments
                        return py::cast(sys.reach(p, a));
                    });
            },
            "reach(initial_set=X0, input_set=U, t_final=T, alg='linearization', time_step=dt, "
            "taylor_terms=4, zonotope_order=50, tensor_order=2, error_order, intermediate_order): "
            "the reachable sets; alg='linearization_adaptive' (zeta_z, ...) chooses the step "
            "sizes itself, and the result has their times")
        // the simulations
        .def(
            "simulate",
            [](const py::object &self, const py::args &args, const py::kwargs &kwargs) {
                const NonlinearSys &sys = py::cast<const NonlinearSys &>(self);
                return run_with_kwargs<SimulationProblem, alg::NonlinearSimulate>(
                    self, "simulate", args, kwargs, {},
                    [&](const SimulationProblem &p, const alg::NonlinearSimulate &a) {
                        return stack_time(sys.simulate(p, a));
                    });
            },
            "simulate(initial_state=x0 (n, N), t_final=T, time_step=dt, max_step=1e-3): the "
            "states (steps + 1, n, N), time first; classical Runge-Kutta")
        .def(
            "simulate_random",
            [](const py::object &self, const py::args &args, const py::kwargs &kwargs) {
                // random simulations of the nonlinear system
                const NonlinearSys &sys = py::cast<const NonlinearSys &>(self);
                const int64_t points =
                    py::cast<int64_t>(required(kwargs, "points", "simulate_random"));
                cora::Rng &rng = py::cast<cora::Rng &>(required(kwargs, "rng", "simulate_random"));
                return run_with_kwargs<RandomSimulationProblem, alg::NonlinearSimulate>(
                    self, "simulate_random", args, kwargs, {"points", "rng"},
                    [&](const RandomSimulationProblem &p, const alg::NonlinearSimulate &a) {
                        return stack_time(sys.simulate_random(p, a, points, rng));
                    });
            },
            "simulate_random(initial_set=X0, t_final=T, time_step=dt, points=N, rng=rng)");
}

/// The differential-algebraic systems: f(x, y, u) and g(x, y, u) as symbolic expressions.
void bind_nonlin_das_sys(py::module_ &m) {
    py::class_<DaeReachSet>(m, "DaeReachSet",
                            "The reachable sets of a differential-algebraic system.")
        .def_readonly("time_step", &DaeReachSet::time_step, "the step between the sets")
        .def_readonly("time_interval", &DaeReachSet::time_interval, "the states over each step")
        .def_readonly("time_point", &DaeReachSet::time_point, "the states at each step time")
        .def_readonly("algebraic", &DaeReachSet::algebraic,
                      "the algebraic variables over each step");
    py::class_<NonlinDASys>(m, "NonlinDASys",
                            "The system x' = f(x, y, u), 0 = g(x, y, u), index 1.")
        .def(py::init([](const py::function &f, const py::function &g, int64_t dim, int64_t inputs,
                         int64_t constraints) {
                 // f and g run once, here, on symbolic states, algebraic variables and inputs
                 const auto rows = [](const py::function &fn) {
                     return [fn](const std::vector<Expr> &x, const std::vector<Expr> &y,
                                 const std::vector<Expr> &u) {
                         return fn(x, y, u).cast<std::vector<Expr>>();
                     };
                 };
                 return NonlinDASys(rows(f), rows(g), dim, inputs, constraints);
             }),
             py::arg("f"), py::arg("g"), py::arg("dim"), py::arg("inputs") = 0,
             py::arg("constraints") = 1,
             // the sizes and the Newton solvers follow
             "f(x, y, u) and g(x, y, u) return the dim and constraints components of the equations")
        // the sizes and the Newton solvers
        .def_property_readonly("dim", &NonlinDASys::dim, "The number of states n")
        .def_property_readonly("input_dim", &NonlinDASys::input_dim, "The number of inputs m")
        .def_property_readonly("constraint_dim", &NonlinDASys::constraint_dim,
                               "The number of algebraic variables")
        .def(
            "consistent_initial_state",
            [](const NonlinDASys &sys, const py::object &x0, const py::object &y0,
               const py::object &u0) {
                // the vectors from Python are columns
                return vector(sys.consistent_initial_state(vector_of(x0, "x0"), vector_of(y0, "y0"),
                                                           vector_of(u0, "u0")));
            },
            py::arg("x0"), py::arg("y0"), py::arg("u0") = py::list(),
            "The algebraic variables with g(x0, y, u0) = 0 by Newton's method from y0")
        .def(
            "steady_state",
            [](const NonlinDASys &sys, const py::object &x0, const py::object &y0,
               const py::object &u0) {
                // the state and y as a pair of vectors
                const auto [x, y] =
                    sys.steady_state(vector_of(x0, "x0"), vector_of(y0, "y0"), vector_of(u0, "u0"));
                return py::make_tuple(vector(x), vector(y));
            },
            py::arg("x0"), py::arg("y0"), py::arg("u0") = py::list(),
            // the steady state returns the state and the algebraic variables
            "The state and the algebraic variables with f = 0 and g = 0 at the input u0")
        // reach and simulate
        .def(
            "reach",
            [](const py::object &self, const py::args &args, const py::kwargs &kwargs) {
                // the problem and the algorithm from the keywords
                const NonlinDASys &sys = py::cast<const NonlinDASys &>(self);
                return run_with_kwargs<DaeReachProblem, alg::DaeReach>(
                    self, "reach", args, kwargs, {},
                    [&](const DaeReachProblem &p, const alg::DaeReach &a) {
                        return py::cast(sys.reach(p, a));
                    });
            },
            "reach(initial_set=X0, input_set=U, u_trans=u (m, steps), y0_guess=y0, t_start=0, "
            "t_final=T, time_step=dt, taylor_terms=4, zonotope_order=50, error_order=3, "
            "max_error_x, max_error_y): the reachable sets")
        .def(
            "simulate",
            [](const NonlinDASys &sys, const py::object &x0, const py::object &y_guess,
               double t_final, double time_step, const py::object &u, double max_step) {
                // the Runge-Kutta samples of x and y
                const DaeTrajectory T = sys.simulate(
                    vector_of(x0, "x0"), vector_of(y_guess, "y_guess"), vector_of(u, "u"), t_final,
                    alg::RungeKutta4{.time_step = time_step, .max_step = max_step});
                return py::make_tuple(matrix(T.x), matrix(T.y));
            },
            py::arg("x0"), py::arg("y_guess"), py::arg("t_final"), py::arg("time_step"),
            py::arg("u") = py::list(), py::arg("max_step") = 1e-3,
            "The states (n, steps + 1) and the algebraic variables (p, steps + 1) of a trajectory");
}

/// The activation of a name: identity, relu, sigmoid or tanh.
Activation activation_of(const std::string &name) {
    if (name == "identity") return Activation::identity;
    if (name == "relu") return Activation::relu;
    if (name == "sigmoid") return Activation::sigmoid;
    if (name == "tanh") return Activation::tanh;
    CORA_FAIL(Error::wrong_value,
              "NeuralNetwork: the activation must be identity, relu, sigmoid or tanh, got " + name);
}

/// The method of the network's line over an activation from its name.
PolyMethod poly_method_of(const std::string &name) {
    if (name != "singh" && name != "regression")
        CORA_FAIL(Error::wrong_value, "poly_method must be 'singh' or 'regression', got " + name);
    return name == "singh" ? PolyMethod::singh : PolyMethod::regression;
}

/// The options of the plant's alg "lin" and of the network from keyword arguments.
ClosedLoopOptions closed_loop_options(const py::kwargs &kwargs) {
    ClosedLoopOptions options;
    options.plant.time_step = py::cast<double>(required(kwargs, "time_step", "reach"));
    const auto read = [&](const char *key, auto &field) {
        if (kwargs.contains(key)) field = py::cast<std::decay_t<decltype(field)>>(kwargs[key]);
    };
    read("taylor_terms", options.plant.taylor_terms);
    read("zonotope_order", options.plant.zonotope_order);
    read("tensor_order", options.plant.tensor_order);
    read("error_order", options.plant.error_order);
    read("intermediate_order", options.plant.intermediate_order);
    read("add_approx_error_to_gi", options.nn.add_approx_error_to_gi);
    if (kwargs.contains("poly_method"))
        options.nn.poly_method = poly_method_of(py::cast<std::string>(kwargs["poly_method"]));
    return options;
}

/// The specification of keyword arguments: spec and the optional spec_time = (t0, t1).
std::optional<TimedSpecification> timed_specification(const py::kwargs &kwargs) {
    if (!kwargs.contains("spec") || kwargs["spec"].is_none()) return std::nullopt;
    TimedSpecification out{py::cast<Specification>(kwargs["spec"])};
    if (kwargs.contains("spec_time") && !kwargs["spec_time"].is_none()) {
        const auto t = py::cast<std::array<double, 2>>(kwargs["spec_time"]);
        out.time = std::make_pair(t[0], t[1]);
    }
    return out;
}

/// The sets of a closed-loop reach as a dict.
template <class S>
py::dict closed_loop_dict(const ClosedLoopReach<S> &R) {
    // the sets and times as lists, the state of the run as flags
    py::dict out;
    out["time_interval"] = py::cast(R.time_interval);
    out["time_point"] = py::cast(R.time_point);
    out["time"] = py::cast(R.time);
    out["completed"] = R.completed;
    out["violated"] = R.violated;
    out["exploded"] = R.exploded;
    return out;
}

void bind_neural_network(py::module_ &m) {
    // the network: layers, activations, and the evaluation on points and sets
    py::class_<NeuralNetwork>(m, "NeuralNetwork",
                              "Affine layers with an activation after each (a ReLU after each "
                              "but the last by default).")
        .def(py::init([](const std::vector<std::pair<py::object, py::object>> &layers,
                         const std::optional<std::vector<std::string>> &activations) {
                 std::vector<NeuralNetwork::Layer> out;
                 out.reserve(layers.size());
                 for (const auto &[W, b] : layers)
                     out.push_back({tensor_of(W, "the weights"), tensor_of(b, "the bias")});
                 if (!activations) return NeuralNetwork(out);
                 // one activation per layer
                 std::vector<Activation> acts;
                 for (const std::string &name : *activations) acts.push_back(activation_of(name));
                 return NeuralNetwork(out, acts);
             }),
             py::arg("layers"), py::arg("activations") = py::none(),
             "the layers as (W (out, in), b (out,)) pairs, first to last, and optionally the "
             "activation of each: 'identity', 'relu', 'sigmoid' or 'tanh'")
        .def_property_readonly("input_dim", &NeuralNetwork::input_dim, "The number of inputs")
        .def_property_readonly("output_dim", &NeuralNetwork::output_dim, "The number of outputs")
        // the outputs at points, the enclosure of the outputs over a set, and the pairs
        .def(
            "evaluate",
            [](const NeuralNetwork &nn, const py::object &x,
               const std::string &poly_method) -> py::object {
                if (py::isinstance<Zonotope>(x))
                    return py::cast(nn.evaluate(py::cast<Zonotope>(x),
                                                {.poly_method = poly_method_of(poly_method)}));
                const Tensor y = nn.evaluate(tensor_of(x, "the points"));
                const bool one = py::hasattr(x, "ndim") && py::cast<int>(x.attr("ndim")) == 1;
                return one ? vector(y) : matrix(y);
            },
            py::arg("x"), py::arg("poly_method") = "singh",
            "The outputs at a point (in,) or the columns of (in, N); for a zonotope, "
            "a zonotope that contains the outputs of every point, the line of each activation by "
            "poly_method 'singh' or 'regression'")
        // the Jacobian at a point
        .def(
            "sensitivity",
            [](const NeuralNetwork &nn, const py::object &x) {
                const Tensor point = tensor_of(x, "the point");
                return matrix(nn.sensitivity(point.reshape({point.numel(), 1})));
            },
            py::arg("x"), "The Jacobian (out, in) of the network at the point x (in,)")
        // the pairs (x, N(x)) of a set, the input rows kept
        .def(
            "evaluate_with_input",
            [](const NeuralNetwork &nn, const py::object &X, bool add_approx_error_to_gi,
               const std::string &poly_method) -> py::object {
                const NnOptions options{.add_approx_error_to_gi = add_approx_error_to_gi,
                                        .poly_method = poly_method_of(poly_method)};
                if (py::isinstance<PolyZonotope>(X))
                    return py::cast(nn.evaluate_with_input(py::cast<PolyZonotope>(X), options));
                return py::cast(nn.evaluate_with_input(py::cast<Zonotope>(X), options));
            },
            py::arg("X"), py::arg("add_approx_error_to_gi") = true,
            py::arg("poly_method") = "singh",
            "The set of the pairs (x, N(x)) of a zonotope or polynomial zonotope X");

    // the neural network controlled system: reach, simulate and verify with keyword options
    py::class_<NeurNetContrSys>(m, "NeurNetContrSys",
                                "A NonlinearSys plant whose first inputs come from a neural "
                                "network that is sampled every dt.")
        .def(py::init<NonlinearSys, NeuralNetwork, double>(), py::arg("plant"), py::arg("nn"),
             py::arg("dt"))
        .def_property_readonly("dim", &NeurNetContrSys::dim, "The number of states of the plant")
        .def_property_readonly("dt", &NeurNetContrSys::dt, "The sampling time")
        // reach from a zonotope or a polynomial zonotope, with the options as keywords
        .def(
            "reach",
            [](const NeurNetContrSys &sys, const py::object &X0, double t_final,
               const py::kwargs &kwargs) -> py::object {
                const std::optional<Zonotope> U =
                    kwargs.contains("input_set") && !kwargs["input_set"].is_none()
                        ? std::optional<Zonotope>(py::cast<Zonotope>(kwargs["input_set"]))
                        : std::nullopt;
                const ClosedLoopOptions options = closed_loop_options(kwargs);
                const std::optional<TimedSpecification> spec = timed_specification(kwargs);
                if (py::isinstance<PolyZonotope>(X0))
                    return closed_loop_dict(
                        sys.reach(py::cast<PolyZonotope>(X0), U, t_final, options, spec));
                return closed_loop_dict(
                    sys.reach(py::cast<Zonotope>(X0), U, t_final, options, spec));
            },
            py::arg("initial_set"), py::arg("t_final"),
            // the keywords and the dict that comes back
            "reach(X0, t_final, time_step=dt, input_set=U, spec=Specification, spec_time=(t0, t1), "
            "taylor_terms=4, zonotope_order=50, tensor_order=2, ...): a dict of the sets over and "
            "at the steps (time_interval, time_point, time) and completed, violated, exploded")
        // runs with the output held over a control period
        .def(
            "simulate",
            [](const NeurNetContrSys &sys, const py::object &x0, double t_final, double time_step,
               const py::object &u) {
                const ClosedLoopTrajectories T = sys.simulate(
                    tensor_of(x0, "x0"),
                    u.is_none() ? std::nullopt : std::optional<Tensor>(tensor_of(u, "u")), t_final,
                    time_step);
                return py::make_tuple(py::cast(T.time), stack_time(T.x));
            },
            py::arg("x0"), py::arg("t_final"), py::arg("time_step"), py::arg("u") = py::none(),
            "The times and the states (steps + 1, n, N) of the runs from the columns x0 (n, N)")
        // CORA's verify: random runs, then the reach, and the verdict as a string
        .def(
            "verify",
            [](const NeurNetContrSys &sys, const py::object &X0, double t_final, int64_t points,
               cora::Rng &rng, const py::kwargs &kwargs) {
                const std::optional<Zonotope> U =
                    kwargs.contains("input_set") && !kwargs["input_set"].is_none()
                        ? std::optional<Zonotope>(py::cast<Zonotope>(kwargs["input_set"]))
                        : std::nullopt;
                const ClosedLoopOptions options = closed_loop_options(kwargs);
                const std::optional<TimedSpecification> spec = timed_specification(kwargs);
                // the specification is required, the verdict of either kind of start set
                if (!spec) CORA_FAIL(Error::wrong_value, "verify: give the specification as spec=");
                VerifySplit split;
                if (kwargs.contains("split_depth"))
                    split.depth = py::cast<int>(kwargs["split_depth"]);
                if (kwargs.contains("split_threads"))
                    split.max_threads = py::cast<int>(kwargs["split_threads"]);
                const ClosedLoopVerdict v = py::isinstance<PolyZonotope>(X0)
                                                ? sys.verify(py::cast<PolyZonotope>(X0), U, t_final,
                                                             options, *spec, points, rng, split)
                                                : sys.verify(py::cast<Zonotope>(X0), U, t_final,
                                                             options, *spec, points, rng, split);
                return v.verdict == Verdict::verified    ? "VERIFIED"
                       : v.verdict == Verdict::falsified ? "FALSIFIED"
                                                         : "UNKNOWN";
            },
            py::arg("initial_set"), py::arg("t_final"), py::arg("points"), py::arg("rng"),
            "verify(X0, t_final, points, rng, time_step=dt, spec=Specification, split_depth=0, "
            "split_threads=8, poly_method='singh', ...): 'VERIFIED', 'FALSIFIED' or 'UNKNOWN' "
            "(CORA's verify with splitR0 = split_depth)");
}

// Plotting ----------------------------------------------------------------------------------------
// The plotting logic is C++ (plot/plot.h): a Figure collects layers, Python draws them.

/// A color from Python: a CORA identifier ("CORA:red") or an RGB triple.
Color color_of(const py::handle &value) {
    if (py::isinstance<py::str>(value)) return Color(py::cast<std::string>(value));
    const std::vector<double> rgb = py::cast<std::vector<double>>(value);
    if (rgb.size() != 3)
        CORA_FAIL(Error::wrong_value, "cora.plot: a color is a CORA identifier such as "
                                      "'CORA:red' or an (r, g, b) triple");
    return Color(rgb[0], rgb[1], rgb[2]);
}

/// The plot options given as keyword arguments, named as in PlotOptions.
PlotOptions options_of(const py::kwargs &kwargs) {
    PlotOptions o;
    const std::vector<std::string> known = {
        "label",  "color", "face_color",  "edge_color", "face_alpha", "line_width", "unify",
        "filled", "set",   "time_points", "step",       "num_colors", "cidx"};
    for (const auto &item : kwargs) {
        const std::string key = py::cast<std::string>(item.first);
        // the options of a layer, by keyword
        if (key == "label")
            o.label = py::cast<std::string>(item.second);
        else if (key == "color")
            o.color = color_of(item.second);
        else if (key == "face_color")
            o.face_color = color_of(item.second);
        else if (key == "edge_color")
            o.edge_color = color_of(item.second);
        else if (key == "face_alpha")
            o.face_alpha = py::cast<double>(item.second);
        else if (key == "line_width")
            o.line_width = py::cast<double>(item.second);
        else if (key == "unify")
            o.unify = py::cast<bool>(item.second);
        else if (key == "filled")
            o.filled = py::cast<bool>(item.second);
        else if (key == "set")
            o.set = py::cast<std::string>(item.second);
        else if (key == "time_points")
            o.time_points = py::cast<bool>(item.second);
        // how the sets are drawn over time and in color
        else if (key == "step")
            o.step = py::cast<int>(item.second);
        else if (key == "num_colors")
            o.num_colors = py::cast<int>(item.second);
        else if (key == "cidx")
            o.cidx = py::cast<int>(item.second);
        else
            CORA_FAIL(Error::redundant_name_value_pair,
                      "cora.plot: unknown option '" + key + "'" + did_you_mean(key, known));
    }
    return o;
}

/// The TikZ options given as keyword arguments, named as in TikzOptions.
TikzOptions tikz_options_of(const py::kwargs &kwargs) {
    TikzOptions o;
    const std::vector<std::string> known = {"standalone",    "coramacs", "thin",  "thin_tolerance",
                                            "external_data", "width",    "height"};
    for (const auto &item : kwargs) {
        const std::string key = py::cast<std::string>(item.first);
        // the options of the TikZ output, by keyword
        if (key == "standalone")
            o.standalone = py::cast<bool>(item.second);
        else if (key == "coramacs")
            o.coramacs = py::cast<bool>(item.second);
        else if (key == "thin")
            o.thin = py::cast<bool>(item.second);
        else if (key == "thin_tolerance")
            o.thin_tolerance = py::cast<double>(item.second);
        else if (key == "external_data")
            o.external_data = py::cast<bool>(item.second);
        else if (key == "width")
            o.width = py::cast<std::string>(item.second);
        else if (key == "height")
            o.height = py::cast<std::string>(item.second);
        else
            CORA_FAIL(Error::redundant_name_value_pair,
                      "to_tikz: unknown option '" + key + "'" + did_you_mean(key, known));
    }
    return o;
}

// the colors as triples
py::tuple rgb(const Color &c) { return py::make_tuple(c.r, c.g, c.b); }

/// A layer of a figure as a dict, for the code that draws it.
py::dict layer_dict(const Figure::Layer &l) {
    py::dict d;
    switch (l.kind) {
    case Figure::Kind::polygons:
        d["kind"] = "polygons";
        break;
    case Figure::Kind::polyline:
        d["kind"] = "polyline";
        break;
    case Figure::Kind::points:
        d["kind"] = "points";
        break;
    case Figure::Kind::region:
        d["kind"] = "region";
        break;
    case Figure::Kind::surface:
        d["kind"] = "surface";
        break;
    }
    py::list polygons, faces;
    // the polygons and faces of a layer as arrays
    for (const Polygon &p : l.polygons) polygons.append(polygon_array(p));
    for (const std::vector<Point3> &f : l.faces) faces.append(points3_array(f));
    d["polygons"] = polygons;
    d["faces"] = faces;
    d["edge"] = rgb(l.edge);
    d["face"] = l.filled || l.kind == Figure::Kind::region ? py::object(rgb(l.face)) : py::none();
    d["unify"] = l.unify;
    d["face_alpha"] = l.face_alpha;
    d["dashed"] = l.dashed;
    d["line_width"] = l.line_width;
    d["radius"] = l.radius;
    d["a"] = py::make_tuple(l.a[0], l.a[1]);
    d["b"] = l.b;
    d["sign"] = l.sign;
    d["label"] = l.label;
    d["zorder"] = l.zorder;
    return d;
}

/// The time points of a simulation (steps, n, N) from torch or numpy as one Tensor (n, N) each.
std::vector<Tensor> trajectories(const py::object &o) {
    std::vector<Tensor> x;
#ifdef CORACPP_PYTHON_TORCH
    py::detail::make_caster<torch::Tensor> torch_caster;
    if (torch_caster.load(o, /*convert=*/false)) {
        for (const torch::Tensor &t : py::cast<torch::Tensor>(o).unbind(0))
            x.push_back(from_torch(t));
        return x;
    }
#endif
    for (const py::handle &step : o) x.push_back(host_tensor(step, false));
    return x;
}

void bind_plot(py::module_ &m) {
    // the figure holds the layers and the axes settings
    py::class_<Figure>(m, "Figure",
                       "The layers that plot puts into one axes, with every choice made.")
        .def(py::init<>())
        .def_readwrite("title", &Figure::title, "The title of the axes")
        .def_readwrite("xlabel", &Figure::xlabel, "The label of the x axis")
        .def_readwrite("ylabel", &Figure::ylabel, "The label of the y axis")
        .def_readwrite("zlabel", &Figure::zlabel, "The label of the z axis")
        .def_readwrite("equal_axes", &Figure::equal_axes,
                       "Whether one unit is as long on every axis")
        .def_readwrite("azimuth", &Figure::azimuth, "The azimuth of a 3-D view in degrees")
        .def_readwrite("elevation", &Figure::elevation, "The elevation of a 3-D view in degrees")
        .def_readwrite("xlim", &Figure::xlim, "The limits of the x axis, or None to fit the layers")
        .def_readwrite("ylim", &Figure::ylim, "The limits of the y axis, or None to fit the layers")
        .def_readwrite("zlim", &Figure::zlim, "The limits of the z axis, or None to fit the layers")
        .def_readonly("dimension", &Figure::dimension, "2, or 3 once a surface is drawn")
        // plot a set in the dims
        .def(
            "plot",
            [](Figure &fig, const ContSet &S, const std::vector<int64_t> &dims,
               const py::kwargs &kwargs) { plot(fig, S, dims, options_of(kwargs)); },
            py::arg("obj"), py::arg("dims"),
            "Draws a set, projected onto the one, two or three dimensions dims")
        .def(
            "plot",
            [](Figure &fig, const ReachSet &R, const std::vector<int64_t> &dims,
               const py::kwargs &kwargs) { plot(fig, R, dims, options_of(kwargs)); },
            py::arg("obj"), py::arg("dims"), "Draws the reachable sets, projected onto dims")
        .def(
            "plot",
            [](Figure &fig, const Specification &spec, const std::vector<int64_t> &dims,
               const py::kwargs &kwargs) { plot(fig, spec, dims, options_of(kwargs)); },
            py::arg("obj"), py::arg("dims"), "Draws the halfspaces of a specification onto dims")
        .def(
            "plot",
            [](Figure &fig, const py::object &data, const std::vector<int64_t> &dims,
               const py::kwargs &kwargs) {
                // Points (n, N) or the time points of a simulation (steps, n, N).
                const py::ssize_t ndim = py::hasattr(data, "ndim")
                                             ? py::cast<py::ssize_t>(data.attr("ndim"))
                                             : py::cast<py::ssize_t>(data.attr("dim")());
                if (ndim == 3)
                    plot(fig, trajectories(data), dims, options_of(kwargs));
                else
                    plot(fig, tensor_of(data, "the points"), dims, options_of(kwargs));
            },
            py::arg("obj"), py::arg("dims"), "Draws points (n, N) or simulations (steps, n, N)")
        .def(
            "plot_over_time",
            [](Figure &fig, const ReachSet &R, int64_t dim, const py::kwargs &kwargs) {
                plot_over_time(fig, R, dim, options_of(kwargs));
            },
            py::arg("obj"), py::arg("dim"), "Draws a dimension of a reachable set over time")
        .def(
            "plot_over_time",
            [](Figure &fig, const py::object &x, int64_t dim, double time_step,
               const py::kwargs &kwargs) {
                // a dimension of simulations over time
                plot_over_time(fig, trajectories(x), dim, time_step, options_of(kwargs));
            },
            py::arg("obj"), py::arg("dim"), py::arg("time_step"),
            "Draws a dimension of simulations (steps, n, N) over time")
        .def(
            "plot_rand_point",
            [](Figure &fig, const ContSet &S, const std::vector<int64_t> &dims, int64_t N, Rng &rng,
               const py::kwargs &kwargs) {
                plot_rand_point(fig, S, dims, N, rng, options_of(kwargs));
            },
            py::arg("obj"), py::arg("dims"), py::arg("num_points"), py::arg("rng"),
            "Draws random points of a set as dots")
        .def(
            "layers",
            [](const Figure &fig) {
                py::list layers;
                for (const Figure::Layer &l : fig.layers()) layers.append(layer_dict(l));
                return layers;
            },
            "All layers drawn, as dicts")
        // the layers are taken out of the figure
        .def(
            "take_layers",
            [](Figure &fig) {
                py::list layers;
                for (const Figure::Layer &l : fig.take_layers()) layers.append(layer_dict(l));
                return layers;
            },
            "The layers drawn since the last call, as dicts, and forgets them")
        .def(
            "limits",
            [](const Figure &fig, double pw, double ph) {
                const Figure::Limits l = fig.limits(pw, ph);
                return py::make_tuple(l.x0, l.x1, l.y0, l.y1, l.z0, l.z1);
            },
            py::arg("pw") = 0.0, py::arg("ph") = 0.0, "The axis limits (x0, x1, y0, y1, z0, z1)")
        .def("svg", &Figure::svg, "The figure as SVG text")
        .def("save", &Figure::save, py::arg("path"), "Writes the figure as an SVG file")
        .def(
            "tikz",
            [](const Figure &fig, const py::kwargs &kwargs) {
                // the figure as pgfplots code
                return fig.tikz(tikz_options_of(kwargs));
            },
            "The figure as pgfplots code")
        .def(
            "save_tikz",
            [](const Figure &fig, const std::string &path, const py::kwargs &kwargs) {
                fig.save_tikz(path, tikz_options_of(kwargs));
            },
            py::arg("path"), "Writes the figure as pgfplots code (and tables) to path")
        .def_static(
            "clip_region",
            [](const py::array_t<double, py::array::c_style | py::array::forcecast> &polygon,
               const std::array<double, 2> &a, double b, double sign) {
                Polygon corners;
                auto view = polygon.unchecked<2>();
                for (py::ssize_t i = 0; i < view.shape(0); ++i)
                    corners.push_back({view(i, 0), view(i, 1)});
                // clip a polygon by a half-plane
                return polygon_array(Figure::clip_region(corners, a, b, sign));
            },
            py::arg("polygon"), py::arg("a"), py::arg("b"), py::arg("sign"),
            "The part of a polygon where sign * (a.x - b) >= 0");

    py::class_<FigureGrid>(m, "FigureGrid",
                           "Figures in rows and columns: SVG cells or a groupplot.")
        .def(py::init<int, int>(), py::arg("rows"), py::arg("cols"))
        .def_readwrite("share_x", &FigureGrid::share_x,
                       "Whether the cells of a column share their x axis")
        .def_readwrite("share_y", &FigureGrid::share_y,
                       "Whether the cells of a row share their y axis")
        .def_readwrite("width", &FigureGrid::width, "The width of the grid in pixels")
        .def_readwrite("height", &FigureGrid::height, "The height of the grid in pixels")
        .def_property_readonly("rows", &FigureGrid::rows, "The number of rows")
        .def_property_readonly("cols", &FigureGrid::cols, "The number of columns")
        .def(
            "at", [](FigureGrid &g, int row, int col) -> Figure & { return g.at(row, col); },
            py::arg("row"), py::arg("col"), py::return_value_policy::reference_internal,
            "The figure in a cell")
        // cells of a grid of figures
        .def(
            "set", [](FigureGrid &g, int row, int col, const Figure &f) { g.at(row, col) = f; },
            py::arg("row"), py::arg("col"), py::arg("figure"),
            "Puts a copy of a figure into a cell")
        .def("svg", &FigureGrid::svg, "The grid as SVG text")
        .def("save", &FigureGrid::save, py::arg("path"), "Writes the grid as an SVG file")
        .def(
            "tikz",
            [](const FigureGrid &g, const py::kwargs &kwargs) {
                return g.tikz(tikz_options_of(kwargs));
            },
            "The grid as one pgfplots groupplot")
        .def(
            "save_tikz",
            [](const FigureGrid &g, const std::string &path, const py::kwargs &kwargs) {
                g.save_tikz(path, tikz_options_of(kwargs));
            },
            py::arg("path"), "Writes the grid as pgfplots code (and tables) to path");

    m.def("use_cora_colors", &use_cora_colors, py::arg("scheme"),
          "'CORA:default' (the next color of CORA's order for every set) or 'CORA:contDynamics'");
    // the CORA colors
    m.def(
        "cora_color",
        [](const std::string &identifier, int num_colors, int cidx, double alpha) {
            return rgb(cora_color(identifier, num_colors, cidx, alpha));
        },
        py::arg("identifier"), py::arg("num_colors") = 1, py::arg("cidx") = 1,
        py::arg("alpha") = 0.2, "The RGB triple of a CORA color such as 'CORA:reachSet'");
    m.def(
        "cora_color",
        [](int number) { return rgb(cora_color("CORA:color" + std::to_string(number))); },
        py::arg("number"), "The RGB triple of the number-th color of CORA's palette");
}

/// The guard intersection method of its Python name, with the enclosure methods.
alg::GuardIntersect hybrid_guard_method(const std::string &name,
                                        const std::vector<std::string> &enclose, int guard_order,
                                        const std::string &box, int fm_passes) {
    // the enclosure methods by their names
    std::vector<alg::Enclose> methods;
    for (const std::string &e : enclose) {
        if (e == "box")
            methods.push_back(alg::Enclose::box);
        else if (e == "pca")
            methods.push_back(alg::Enclose::pca);
        else if (e == "flow")
            methods.push_back(alg::Enclose::flow);
        else
            CORA_FAIL(Error::wrong_value,
                      "reach: enclose is 'box', 'pca' or 'flow', got '" + e + "'");
    }
    // how a cut by a polytope guard is boxed
    alg::CutBox boxing = alg::CutBox::fm;
    if (box == "cert")
        boxing = alg::CutBox::cert;
    else if (box == "both")
        boxing = alg::CutBox::both;
    else if (box != "fm")
        CORA_FAIL(Error::wrong_value, "reach: box is 'fm', 'cert' or 'both', got '" + box + "'");
    // the guard intersection by its name
    if (name == "zono_girard") return alg::ZonoGirard{methods};
    if (name == "nondet_guard") return alg::NondetGuard{methods, boxing, fm_passes};
    if (name == "con_zonotope")
        return alg::ConZonotopeGuard{methods, guard_order, boxing, fm_passes};
    CORA_FAIL(Error::wrong_value, "reach: guard_intersect is 'zono_girard', 'nondet_guard' or "
                                  "'con_zonotope', got '" +
                                      name + "'");
}

/// Sets as a list of their Python classes.
py::list hybrid_sets(const std::vector<Set> &sets) {
    // each set as an object of its own class
    py::list out;
    for (const Set &S : sets) out.append(object_of(S));
    return out;
}

void bind_lin_param_sys(py::module_ &m) {
    // the system, with B and c optional
    py::class_<LinParamSys>(m, "LinParamSys",
                            "The linear system x' = A x + B u + c, A in a matrix zonotope.")
        .def(py::init([](const MatZonotope &A, const py::object &B, const py::object &c,
                         const std::string &type) {
                 const auto opt = [](const py::object &o, const char *what) {
                     return o.is_none() ? std::nullopt : std::optional<Tensor>(tensor_of(o, what));
                 };
                 // constant or varying parameters, as CORA's 'constParam' and 'varParam'
                 if (type != "constant" && type != "varying")
                     CORA_FAIL(Error::wrong_value,
                               "LinParamSys: type is 'constant' or 'varying', got '" + type + "'");
                 return LinParamSys(A, opt(B, "B"), opt(c, "c"),
                                    type == "constant" ? ParamType::constant : ParamType::varying);
             }),
             py::arg("A"), py::arg("B") = py::none(), py::arg("c") = py::none(),
             py::arg("type") = "constant",
             "A a MatZonotope (n, n), B (n, m) and c (n, 1) optional; type 'constant' or 'varying'")
        // its parts, and reach
        .def_property_readonly("A", &LinParamSys::A, "The matrix zonotope of the system matrices")
        .def_property_readonly("dim", &LinParamSys::dim, "The number of states n")
        .def_property_readonly(
            "type",
            [](const LinParamSys &sys) {
                return std::string(sys.type() == ParamType::constant ? "constant" : "varying");
            },
            "'constant' or 'varying' parameters")
        .def(
            "reach",
            [](const py::object &self, const py::args &args, const py::kwargs &kwargs) {
                const LinParamSys &sys = py::cast<const LinParamSys &>(self);
                return run_with_kwargs<ReachProblem, alg::LinParamReach>(
                    self, "reach", args, kwargs, {},
                    [&](const ReachProblem &p, const alg::LinParamReach &a) {
                        // reach with the problem and the algorithm as keywords
                        return py::cast(sys.reach(p, a));
                    });
            },
            "reach(initial_set=X0, t_final=T, time_step=dt, taylor_terms=4, zonotope_order=50, "
            "intermediate_terms=2, comp_time_point=True, input_set=U): the reachable sets");
}

void bind_hybrid(py::module_ &m) {
    // the reset of a transition
    py::class_<LinearReset>(m, "LinearReset", "The reset x+ = A x + B u + c of a transition.")
        .def(py::init([](const Array &A, const py::object &B, const py::object &c) {
                 const auto opt = [](const py::object &o,
                                     const char *what) -> std::optional<Tensor> {
                     if (o.is_none()) return std::nullopt;
                     return tensor_of(o, what);
                 };
                 return LinearReset(tensor_of(A, "A"), opt(B, "B"), opt(c, "c"));
             }),
             py::arg("A"), py::arg("B") = py::none(), py::arg("c") = py::none(),
             "x+ = A x + B u + c: A (n+, n), B (n+, m) and c (n+, 1) optional")
        .def_property_readonly(
            "A", [](const LinearReset &r) { return matrix(r.A()); }, "The state map A")
        .def(
            "evaluate",
            [](const LinearReset &r, const py::object &X, const std::optional<Zonotope> &U) {
                // a set maps to a set, points (n, N) to points
                if (py::isinstance<ContSet>(X)) return object_of(r.evaluate(*set_of(X), U));
                return py::object(matrix(r.evaluate(tensor_of(X, "x"))));
            },
            py::arg("X"), py::arg("U") = py::none(),
            "The image A X + B U + c of a zonotope or a bundle, or of points (n, N)");
    py::class_<Transition>(m, "Transition", "A jump: guard, reset and target location.")
        .def(py::init([](const py::object &guard, const LinearReset &reset, int64_t target) {
                 // a box guard is held as its polytope
                 if (py::isinstance<Interval>(guard))
                     return Transition(py::cast<Interval>(guard), reset, target);
                 if (py::isinstance<LevelSet>(guard))
                     return Transition(py::cast<LevelSet>(guard), reset, target);
                 return Transition(py::cast<Polytope>(guard), reset, target);
             }),
             py::arg("guard"), py::arg("reset"), py::arg("target"),
             "The jump to location `target` (0-based) through the guard (a Polytope, an Interval "
             "or a "
             "LevelSet)")
        .def_property_readonly("guard", &Transition::guard, "The guard: a polytope or a level set")
        .def_property_readonly("reset", &Transition::reset, "The reset")
        .def_property_readonly("target", &Transition::target, "The target location, 0-based");
    // the location, with its flow and invariant (a polytope or a level set)
    py::class_<Location>(m, "Location", "A location: invariant, transitions and flow.")
        .def(py::init([](const std::string &name, const py::object &inv,
                         const std::vector<Transition> &trans, const LinearSys &flow,
                         const py::object &c) {
                 return Location(name, region_of(inv), trans, flow,
                                 c.is_none() ? std::nullopt
                                             : std::optional<Tensor>(tensor_of(c, "c")));
             }),
             py::arg("name"), py::arg("invariant"), py::arg("transitions"), py::arg("flow"),
             py::arg("c") = py::none(), "x' = A x + B u + c of the flow inside the invariant")
        // a parametric flow and a nonlinear flow take the invariant too
        .def(py::init([](const std::string &name, const py::object &inv,
                         const std::vector<Transition> &trans, const LinParamSys &flow) {
                 return Location(name, region_of(inv), trans, flow);
             }),
             py::arg("name"), py::arg("invariant"), py::arg("transitions"), py::arg("flow"),
             "x' = A x + B u + c of a LinParamSys flow (its own c) inside the invariant")
        .def(py::init([](const std::string &name, const py::object &inv,
                         const std::vector<Transition> &trans, const NonlinearSys &flow) {
                 return Location(name, region_of(inv), trans, flow);
             }),
             py::arg("name"), py::arg("invariant"), py::arg("transitions"), py::arg("flow"),
             "x' = f(x, u) of a NonlinearSys flow inside the invariant")
        .def_property_readonly("name", &Location::name, "The name")
        .def_property_readonly("invariant", &Location::invariant,
                               "The invariant: a polytope or a level set")
        .def_property_readonly("transitions", &Location::transitions, "The transitions")
        .def_property_readonly("dim", &Location::dim, "The number of states");
    // the results: one branch per visit of a location
    py::class_<LocationReachSet>(m, "LocationReachSet", "The sets of one visit of a location.")
        .def_readonly("location", &LocationReachSet::location, "The location, 0-based")
        .def_readonly("parent", &LocationReachSet::parent, "The branch that jumped here, -1 none")
        .def_property_readonly(
            "time_point", [](const LocationReachSet &R) { return hybrid_sets(R.time_point); },
            "The sets at the step times")
        .def_property_readonly(
            "time_interval", [](const LocationReachSet &R) { return hybrid_sets(R.time_interval); },
            "The sets over the steps")
        .def_readonly("time_point_times", &LocationReachSet::time_point_times,
                      "The times [lower, upper] of the time-point sets")
        .def_readonly("time_interval_times", &LocationReachSet::time_interval_times,
                      "The times [lower, upper] of the time-interval sets");
    py::class_<HybridReachSet>(m, "HybridReachSet", "The branches of reach and its verdict.")
        .def_readonly("branches", &HybridReachSet::branches, "One per visit of a location")
        .def_readonly("verified", &HybridReachSet::verified, "Whether every specification held");
    // the automaton, and reach with the problem and the algorithm as keywords
    py::class_<HybridAutomaton>(m, "HybridAutomaton", "A hybrid automaton of linear locations.")
        .def(py::init<std::string, std::vector<Location>>(), py::arg("name"), py::arg("locations"),
             "The automaton; the locations are numbered from 0")
        .def_property_readonly("name", &HybridAutomaton::name, "The name")
        .def_property_readonly("locations", &HybridAutomaton::locations, "The locations")
        .def(
            "reach",
            [](const HybridAutomaton &HA, const Zonotope &initial_set, double t_final,
               const py::object &time_step, int64_t start_location, double t_start,
               const py::object &input_set, const std::vector<int64_t> &final_locations,
               int taylor_terms, int zonotope_order, int intermediate_terms,
               const std::string &guard_intersect, const std::vector<std::string> &enclose,
               int guard_order, const std::string &box, int fm_passes, const py::list &specs) {
                HybridReachProblem p;
                p.initial_set = initial_set;
                p.t_final = t_final;
                p.start_location = start_location;
                p.t_start = t_start;
                p.final_locations = final_locations;
                // one input set, or a list of one per location
                if (py::isinstance<Zonotope>(input_set))
                    p.input_set = {py::cast<Zonotope>(input_set)};
                else if (!input_set.is_none())
                    p.input_set = py::cast<std::vector<Zonotope>>(input_set);
                alg::HybridStandard a;
                a.time_step = is_number(time_step)
                                  ? std::vector<double>{py::cast<double>(time_step)}
                                  : py::cast<std::vector<double>>(time_step);
                a.taylor_terms = taylor_terms;
                a.zonotope_order = zonotope_order;
                a.intermediate_terms = intermediate_terms;
                a.guard_intersect =
                    hybrid_guard_method(guard_intersect, enclose, guard_order, box, fm_passes);
                // a specification, or (specification, locations)
                std::vector<HybridSpecification> hs;
                for (const py::handle &s : specs) {
                    if (py::isinstance<Specification>(s))
                        hs.push_back({py::cast<Specification>(s), {}});
                    else
                        hs.push_back({py::cast<Specification>(s[py::int_(0)]),
                                      py::cast<std::vector<int64_t>>(s[py::int_(1)])});
                }
                return HA.reach(p, a, hs);
            },
            // the keywords, with the defaults of HybridReachProblem and alg::HybridStandard
            py::arg("initial_set"), py::arg("t_final"), py::arg("time_step"),
            // where and when the analysis starts, and what it takes as input
            py::arg("start_location") = 0, py::arg("t_start") = 0.0,
            py::arg("input_set") = py::none(), py::arg("final_locations") = std::vector<int64_t>(),
            py::arg("taylor_terms") = 10, py::arg("zonotope_order") = 50,
            py::arg("intermediate_terms") = 2, py::arg("guard_intersect") = "zono_girard",
            py::arg("enclose") = std::vector<std::string>{"box"}, py::arg("guard_order") = 0,
            py::arg("box") = "fm", py::arg("fm_passes") = 10, py::arg("specs") = py::list(),
            "The reachable sets up to t_final (alg::HybridStandard); specs are Specifications or "
            "(Specification, [locations]) pairs");
}

void bind_specification(py::module_ &m) {
    py::class_<Specification>(m, "Specification",
                              "A set the reachable sets must lie in (safe_set) or must not touch "
                              "(unsafe_set), typically a Polytope.")
        // the set is of any class; the set classes decide
        .def_static(
            "safe_set", [](const py::object &S) { return Specification::safe_set(set_of(S)); },
            py::arg("S"), "A safe set: the reachable sets must lie in the set S")
        .def_static(
            "unsafe_set", [](const py::object &S) { return Specification::unsafe_set(set_of(S)); },
            py::arg("S"), "An unsafe set: the reachable sets must not touch the set S")
        .def_static(
            "custom",
            [](const py::function &holds) {
                // the predicate gets the set as its Python class, under the GIL
                return Specification::custom([holds](const ContSet &S) {
                    py::gil_scoped_acquire gil;
                    return py::cast<bool>(holds(py::cast(&S, py::return_value_policy::reference)));
                });
            },
            py::arg("holds"),
            // the predicate is the whole specification
            "A predicate on the set: holds(S) is True where the specification holds")
        .def_property_readonly(
            "type",
            [](const Specification &spec) {
                if (spec.type() == SpecType::custom) return "custom";
                return spec.type() == SpecType::safe_set ? "safe_set" : "unsafe_set";
            },
            "'safe_set', 'unsafe_set' or 'custom'")
        .def_property_readonly(
            "set", [](const Specification &spec) { return object_of(spec.set()); },
            "The set of a safe or unsafe specification")
        .def(
            "check", [](const Specification &spec, const ContSet &S) { return spec.check(S); },
            py::arg("S"), "Whether the set satisfies the specification")
        .def(
            "check",
            [](const Specification &spec, const std::vector<Zonotope> &sets) {
                // whether every set of a list satisfies the specification
                return spec.check(sets);
            },
            py::arg("sets"), "Whether every set of a list (such as R.time_interval) does")
        .def(
            "first_violation",
            [](const Specification &spec, const std::vector<Zonotope> &sets) {
                return spec.first_violation(sets);
            },
            py::arg("sets"), "The index of the first violating set, or -1");
}

/// The oracle of CORALean as `cora.lean`: the same names as cora::lean; vectors are 1-D arrays.
void bind_lean(py::module_ &top) {
    using DType = PyScope<lean::DTypeScope, std::string>;
    py::module_ m =
        top.def_submodule("lean", "Sound float sets computed by CORALean (needs CORACPP_ORACLE)");
    bind_scope<lean::DTypeScope, std::string>(m, "_DTypeScope", "A dtype in a with block");
    m.def(
        "dtype", [](const std::string &spec) { return new DType{spec, std::nullopt}; },
        py::arg("spec"),
        "with cora.lean.dtype('ieee:binary32'): new lean objects in that dtype in the block; "
        "'binary64', 'ieee:<format>', 'dyadic:<p>', 'fixedpoint:<f>', 'float', 'nearest'");
    m.def("set_default_dtype", &lean::set_default_dtype, py::arg("spec"),
          "The dtype of threads without a cora.lean.dtype block");
    m.def("current_dtype", &lean::dtype, "This thread's dtype");
    m.def("serves", &lean::serves, py::arg("op"),
          "Whether the running oracle serves the op, e.g. 'zonotope.enlarge' (an older one does "
          "not)");

    using namespace pybind11::literals;
    using LI = lean::Interval;
    using LZ = lean::Zonotope;
    using LT = lean::Tensor;
    // an array converts to a lean tensor in the current dtype
    const auto tensor = [](const py::handle &x) { return LT::from(tensor_of(x, "x")); };

    // an array (or list) is a lean tensor wherever an operation takes one, in the current dtype
    py::class_<LT>(m, "Tensor", "A matrix of exact values in a dtype")
        .def(py::init(tensor), py::arg("x"), "Exact, or an error if a value is not representable")
        .def_static(
            "from_array", [](const py::object &x) { return LT::from(host_tensor(x, true)); },
            py::arg("x"), "Exact, or an error if a value is not representable in the dtype")
        .def(
            "gather", [](const LT &T) { return bounds(T.gather()); },
            "The values as an array (a vector if one column); an error if one is not a double")
        .def(
            "round_to",
            [](const LT &T, const std::string &dtype, const std::string &mode) {
                auto [value, error] = T.round_to(dtype, mode);
                return std::make_pair(value, error);
            },
            py::arg("dtype"), py::arg("mode") = "nearest",
            "(value, error): the rounded values and an interval enclosing x - value")
        .def_property_readonly("dtype", &LT::dtype, "The dtype of the values");
    py::implicitly_convertible<py::array, LT>();
    py::implicitly_convertible<py::list, LT>();

    // an answer of the oracle with its proof flag
    py::class_<lean::Certified>(m, "Certified", "An answer of the oracle, a proof if `proved`")
        .def_readonly("value", &lean::Certified::value,
                      "The answer, as CORA's operation would give it")
        .def_readonly("proved", &lean::Certified::proved,
                      "True if the oracle proved the value; else it found no proof");
    py::class_<lean::Vertices>(m, "Vertices", "The vertices V (columns) and an error box E")
        .def_readonly("V", &lean::Vertices::V, "The vertices as columns")
        .def_readonly("E", &lean::Vertices::E, "The box that encloses the error of the vertices");

    py::class_<LI>(m, "Interval", "A box of lean values and the oracle's operations on it")
        .def(py::init<LT, LT>(), "inf"_a, "sup"_a)
        .def(py::init<const cora::Interval &>(), "I"_a, "The exact box of a cora.Interval")
        .def_static(
            "from_arrays",
            [](const py::object &inf, const py::object &sup) {
                return LI::from(host_tensor(inf, true), host_tensor(sup, true));
            },
            "inf"_a, "sup"_a, "CORALean's interval.from_arrays, run by the oracle")
        .def(
            "gather",
            [](const LI &I) {
                // the bounds as an interval
                auto [inf, sup] = I.gather();
                return Interval(inf, sup);
            },
            "A cora.Interval; an error if a bound is not a double")
        .def_property_readonly("inf", &LI::inf, "The lower bounds, exact lean values")
        .def_property_readonly("sup", &LI::sup, "The upper bounds, exact lean values")
        .def_static("empty", &LI::empty, "n"_a, "CORALean's interval.empty, run by the oracle")
        .def_static("origin", &LI::origin, "n"_a, "CORALean's interval.origin, run by the oracle")
        .def_static("unbounded", &LI::unbounded, "n"_a,
                    "CORALean's interval.unbounded, run by the oracle")
        .def_static("enclose_points", &LI::enclose_points, "points"_a,
                    "CORALean's interval.enclose_points, run by the oracle")
        .def_static("interval_matrix", &LI::interval_matrix, "c"_a, "r"_a,
                    "CORALean's interval.interval_matrix, run by the oracle")
        .def("abs", &LI::abs, "CORALean's interval.abs, run by the oracle")
        .def("uminus", &LI::uminus, "CORALean's interval.uminus, run by the oracle")
        .def("uplus", &LI::uplus, "CORALean's interval.uplus, run by the oracle")
        .def("sign", &LI::sign, "CORALean's interval.sign, run by the oracle")
        .def("round", &LI::round, "CORALean's interval.round, run by the oracle")
        .def("sqrt", &LI::sqrt, "p"_a = 20, "CORALean's interval.sqrt, run by the oracle")
        // elementary functions of the oracle's interval
        .def("exp", &LI::exp, "p"_a = 20, "CORALean's interval.exp, run by the oracle")
        .def("sin", &LI::sin, "p"_a = 20, "CORALean's interval.sin, run by the oracle")
        .def("cos", &LI::cos, "p"_a = 20, "CORALean's interval.cos, run by the oracle")
        .def("tan", &LI::tan, "p"_a = 20, "CORALean's interval.tan, run by the oracle")
        .def("sinh", &LI::sinh, "p"_a = 20, "CORALean's interval.sinh, run by the oracle")
        .def("cosh", &LI::cosh, "p"_a = 20, "CORALean's interval.cosh, run by the oracle")
        .def("tanh", &LI::tanh, "p"_a = 20, "CORALean's interval.tanh, run by the oracle")
        .def("log", &LI::log, "p"_a = 20, "k"_a = 20, "CORALean's interval.log, run by the oracle")
        .def("asin", &LI::asin, "p"_a = 20, "k"_a = 20,
             "CORALean's interval.asin, run by the oracle")
        .def("acos", &LI::acos, "p"_a = 20, "k"_a = 20,
             "CORALean's interval.acos, run by the oracle")
        .def("atan", &LI::atan, "p"_a = 20, "k"_a = 20,
             "CORALean's interval.atan, run by the oracle")
        .def("asinh", &LI::asinh, "p"_a = 20, "k"_a = 20,
             "CORALean's interval.asinh, run by the oracle")
        .def("acosh", &LI::acosh, "p"_a = 20, "k"_a = 20,
             "CORALean's interval.acosh, run by the oracle")
        .def("atanh", &LI::atanh, "p"_a = 20, "k"_a = 20,
             "CORALean's interval.atanh, run by the oracle")
        // inverse hyperbolic functions and atan2 of the oracle's interval
        .def("atan2", &LI::atan2, "I2"_a, "p"_a = 20, "k"_a = 20,
             "CORALean's interval.atan2, run by the oracle")
        .def("enlarge", &LI::enlarge, "factor"_a, "CORALean's interval.enlarge, run by the oracle")
        .def("power", &LI::power, "exponent"_a, "CORALean's interval.power, run by the oracle")
        .def("rdivide", &LI::rdivide, "I2"_a, "CORALean's interval.rdivide, run by the oracle")
        .def("mrdivide", &LI::mrdivide, "d"_a, "CORALean's interval.mrdivide, run by the oracle")
        .def("plus", &LI::plus, "I2"_a, "CORALean's interval.plus, run by the oracle")
        .def("minus", &LI::minus, "I2"_a, "CORALean's interval.minus, run by the oracle")
        .def("times", &LI::times, "I2"_a, "CORALean's interval.times, run by the oracle")
        .def("unite", &LI::unite, "I2"_a, "CORALean's interval.unite, run by the oracle")
        .def("convex_hull", &LI::convex_hull, "I2"_a,
             "CORALean's interval.convex_hull, run by the oracle")
        .def("mink_diff", &LI::mink_diff, "I2"_a,
             "CORALean's interval.mink_diff, run by the oracle")
        .def("min", &LI::min, "I2"_a, "CORALean's interval.min, run by the oracle")
        .def("max", &LI::max, "I2"_a, "CORALean's interval.max, run by the oracle")
        .def("intersect", &LI::intersect, "I2"_a,
             "CORALean's interval.intersect, run by the oracle")
        .def("enclose", &LI::enclose, "I2"_a, "CORALean's interval.enclose, run by the oracle")
        // set operations of the oracle's interval
        .def("cart_prod", &LI::cart_prod, "I2"_a,
             "CORALean's interval.cart_prod, run by the oracle")
        .def("linear_map", py::overload_cast<const LT &>(&LI::linear_map, py::const_), "M"_a,
             "CORALean's interval.linear_map, run by the oracle")
        .def("linear_map", py::overload_cast<const LI &>(&LI::linear_map, py::const_), "M"_a,
             "CORALean's interval.linear_map, run by the oracle")
        .def("quad_map", &LI::quad_map, "Q"_a, "CORALean's interval.quad_map, run by the oracle")
        .def("le", &LI::le, "I2"_a, "tol"_a = 0.0, "CORALean's interval.le, run by the oracle")
        .def("lt", &LI::lt, "I2"_a, "tol"_a = 0.0, "CORALean's interval.lt, run by the oracle")
        .def("isequal", &LI::isequal, "I2"_a, "tol"_a = 0.0,
             "CORALean's interval.isequal, run by the oracle")
        .def("is_full_dim", &LI::is_full_dim, "tol"_a = 0.0,
             "CORALean's interval.is_full_dim, run by the oracle")
        .def("is_bounded", &LI::is_bounded, "tol"_a = 0.0,
             "CORALean's interval.is_bounded, run by the oracle")
        .def("representsa", &LI::representsa, "type"_a, "tol"_a = 0.0,
             "CORALean's interval.representsa, run by the oracle")
        .def("contains", py::overload_cast<const LI &, double>(&LI::contains, py::const_), "I2"_a,
             "tol"_a = 0.0, "CORALean's interval.contains, run by the oracle")
        // containment and intersection in the oracle
        .def("contains", py::overload_cast<const LZ &, double>(&LI::contains, py::const_), "Z"_a,
             "tol"_a = 0.0, "CORALean's interval.contains, run by the oracle")
        .def("is_intersecting",
             py::overload_cast<const LI &, double>(&LI::is_intersecting, py::const_), "I2"_a,
             "tol"_a = 0.0, "CORALean's interval.is_intersecting, run by the oracle")
        .def("is_intersecting",
             py::overload_cast<const LZ &, double>(&LI::is_intersecting, py::const_), "Z"_a,
             "tol"_a = 0.0, "CORALean's interval.is_intersecting, run by the oracle")
        .def("center", &LI::center, "CORALean's interval.center, run by the oracle")
        .def("rad", &LI::rad, "CORALean's interval.rad, run by the oracle")
        .def("infimum", &LI::infimum, "CORALean's interval.infimum, run by the oracle")
        .def("supremum", &LI::supremum, "CORALean's interval.supremum, run by the oracle")
        .def("radius", &LI::radius, "type"_a = 2, "mode"_a = "ub",
             "CORALean's interval.radius, run by the oracle")
        .def("norm", &LI::norm, "type"_a = 2, "mode"_a = "ub",
             "CORALean's interval.norm, run by the oracle")
        .def("volume", &LI::volume, "CORALean's interval.volume, run by the oracle")
        .def("support_function", &LI::support_function, "dir"_a, "type"_a = "upper",
             "CORALean's interval.support_function, run by the oracle")
        .def("project", &LI::project, "dims"_a, "CORALean's interval.project, run by the oracle")
        // projections in the oracle
        .def("lift", &LI::lift, "N"_a, "dims"_a, "CORALean's interval.lift, run by the oracle")
        .def("project_high_dim", &LI::project_high_dim, "N"_a, "dims"_a,
             "CORALean's interval.project_high_dim, run by the oracle")
        .def("split", &LI::split, "dim"_a, "CORALean's interval.split, run by the oracle")
        .def("partition", &LI::partition, "splits"_a, "ind"_a,
             "CORALean's interval.partition, run by the oracle")
        .def("dirac", &LI::dirac, "n"_a, "CORALean's interval.dirac, run by the oracle")
        .def("boundary_point", &LI::boundary_point, "dir"_a, "start"_a,
             "CORALean's interval.boundary_point, run by the oracle")
        .def("transpose", &LI::transpose, "CORALean's interval.transpose, run by the oracle")
        .def("ctranspose", &LI::ctranspose, "CORALean's interval.ctranspose, run by the oracle")
        .def("prod", &LI::prod, "CORALean's interval.prod, run by the oracle")
        .def("diag", &LI::diag, "CORALean's interval.diag, run by the oracle")
        .def("kron", &LI::kron, "I2"_a, "CORALean's interval.kron, run by the oracle")
        .def("horzcat", &LI::horzcat, "I2"_a, "CORALean's interval.horzcat, run by the oracle")
        .def("vertcat", &LI::vertcat, "I2"_a, "CORALean's interval.vertcat, run by the oracle")
        .def("sum", &LI::sum, "dim"_a, "CORALean's interval.sum, run by the oracle")
        .def("cat", &LI::cat, "I2"_a, "dim"_a, "CORALean's interval.cat, run by the oracle")
        .def("mpower", &LI::mpower, "exponent"_a, "CORALean's interval.mpower, run by the oracle")
        .def("tril", &LI::tril, "K"_a = 0, "CORALean's interval.tril, run by the oracle")
        // triangular parts and reductions
        .def("triu", &LI::triu, "K"_a = 0, "CORALean's interval.triu, run by the oracle")
        .def("reduce", &LI::reduce, "order"_a, "method"_a = "girard",
             "CORALean's interval.reduce, run by the oracle")
        .def("zonotope", &LI::zonotope, "CORALean's interval.zonotope, run by the oracle")
        .def("vertices", &LI::vertices, "CORALean's interval.vertices, run by the oracle");

    using LM = lean::IntervalMatrix;
    py::class_<lean::MatrixVertices>(m, "MatrixVertices",
                                     "The vertex matrices V and an error box E")
        .def_readonly("V", &lean::MatrixVertices::V,
                      "The vertex matrices, one (rows, cols) array each")
        .def_readonly("E", &lean::MatrixVertices::E, "The error box around the vertices");
    py::class_<LM>(m, "IntervalMatrix",
                   "An interval matrix of lean values and the oracle's operations")
        .def(py::init<LI>(), "bounds"_a)
        .def_property_readonly("interval", &LM::interval,
                               "CORALean's intervalMatrix.interval, run by the oracle")
        .def("abs", &LM::abs, "CORALean's intervalMatrix.abs, run by the oracle")
        .def("infimum", &LM::infimum, "CORALean's intervalMatrix.infimum, run by the oracle")
        .def("supremum", &LM::supremum, "CORALean's intervalMatrix.supremum, run by the oracle")
        .def("rad", &LM::rad, "CORALean's intervalMatrix.rad, run by the oracle")
        // the oracle's interval matrix
        .def("plus", &LM::plus, "B"_a, "CORALean's intervalMatrix.plus, run by the oracle")
        .def("representsa", &LM::representsa, "type"_a, "tol"_a = 0.0,
             "CORALean's intervalMatrix.representsa, run by the oracle")
        .def("vertices", &LM::vertices, "CORALean's intervalMatrix.vertices, run by the oracle")
        .def("volume", &LM::volume, "CORALean's intervalMatrix.volume, run by the oracle")
        .def("powers", &LM::powers, "P"_a, "k"_a,
             "CORALean's intervalMatrix.powers, run by the oracle")
        .def("center", &LM::center, "CORALean's intervalMatrix.center, run by the oracle")
        .def("contains", &LM::contains, "M"_a, "tol"_a = 0.0,
             "CORALean's intervalMatrix.contains, run by the oracle")
        .def("exact_square", &LM::exact_square,
             "CORALean's intervalMatrix.exact_square, run by the oracle")
        .def("dependent_terms", &LM::dependent_terms, "r"_a,
             "CORALean's intervalMatrix.dependent_terms, run by the oracle")
        .def("matrix_power", &LM::matrix_power, "exponent"_a,
             "CORALean's intervalMatrix.matrix_power, run by the oracle")
        .def("matmul", &LM::matmul, "B"_a, "CORALean's intervalMatrix.matmul, run by the oracle")
        .def("infinity_norm", &LM::infinity_norm,
             "CORALean's intervalMatrix.infinity_norm, run by the oracle")
        // norms of the oracle's interval matrix
        .def("norm", &LM::norm, "type"_a, "steps"_a = 20,
             "CORALean's intervalMatrix.norm, run by the oracle");

    using LE = lean::EmptySet;
    py::class_<LE>(m, "EmptySet", "The empty set of n dimensions and the oracle's operations on it")
        .def(py::init<int64_t>(), "n"_a)
        .def_property_readonly("dim", &LE::dim, "The number of dimensions n")
        .def_static("empty", &LE::empty, "n"_a, "CORALean's emptyset.empty, run by the oracle")
        .def("intersect", &LE::intersect, "S"_a, "CORALean's emptyset.intersect, run by the oracle")
        .def("plus", &LE::plus, "S"_a, "CORALean's emptyset.plus, run by the oracle")
        .def("convex_hull", py::overload_cast<const LI &>(&LE::convex_hull, py::const_), "I"_a,
             "CORALean's emptyset.convex_hull, run by the oracle")
        .def("convex_hull", py::overload_cast<const LZ &>(&LE::convex_hull, py::const_), "Z"_a,
             "CORALean's emptyset.convex_hull, run by the oracle")
        .def("contains", &LE::contains, "S"_a, "tol"_a = 0.0,
             "CORALean's emptyset.contains, run by the oracle")
        .def("is_intersecting", &LE::is_intersecting, "S"_a, "tol"_a = 0.0,
             "CORALean's emptyset.is_intersecting, run by the oracle")
        .def("isequal", &LE::isequal, "S"_a, "tol"_a = 0.0,
             "CORALean's emptyset.isequal, run by the oracle")
        // predicates of the oracle's empty set
        .def("is_bounded", &LE::is_bounded, "tol"_a = 0.0,
             "CORALean's emptyset.is_bounded, run by the oracle")
        .def("is_full_dim", &LE::is_full_dim, "tol"_a = 0.0,
             "CORALean's emptyset.is_full_dim, run by the oracle")
        .def("center", &LE::center, "The center: None, the set has no point")
        .def("interval", &LE::interval, "CORALean's emptyset.interval, run by the oracle")
        .def("vertices", &LE::vertices, "CORALean's emptyset.vertices, run by the oracle")
        .def("lift", &LE::lift, "N"_a, "dims"_a, "CORALean's emptyset.lift, run by the oracle")
        .def("project", &LE::project, "dims"_a, "CORALean's emptyset.project, run by the oracle")
        .def("project_high_dim", &LE::project_high_dim, "N"_a, "proj"_a,
             "CORALean's emptyset.project_high_dim, run by the oracle")
        .def("radius", &LE::radius, "CORALean's emptyset.radius, run by the oracle")
        .def("volume", &LE::volume, "CORALean's emptyset.volume, run by the oracle")
        .def("complement", &LE::complement, "CORALean's emptyset.complement, run by the oracle");

    py::class_<LZ>(m, "Zonotope",
                   "A nominal zonotope plus an error box, and the oracle's operations")
        .def(py::init([](const Array &c, const Array &G) {
                 return LZ(host_tensor(c, true), host_tensor(G, true));
             }),
             "center"_a, "generators"_a, "An exact zonotope in the current dtype")
        // the oracle's zonotope
        .def(py::init<const Zonotope &>(), "Z"_a, "The exact zonotope of a cora.Zonotope")
        .def("nominal", &LZ::nominal, "c and G with an empty error box")
        .def("error", &LZ::error, "The error box")
        .def_property_readonly("center", &LZ::center, "The nominal center, exact lean values")
        .def_property_readonly("generators", &LZ::generators,
                               "The nominal generators, exact lean values")
        .def("enclosure", &LZ::enclosure, "nominal and error as one zonotope")
        .def("round_to", &LZ::round_to, "dtype"_a, "mode"_a = "nearest",
             "CORALean's zonotope.round_to, run by the oracle")
        .def("linear_map", py::overload_cast<const LT &>(&LZ::linear_map, py::const_), "M"_a,
             "CORALean's zonotope.linear_map, run by the oracle")
        .def("linear_map", py::overload_cast<const LI &>(&LZ::linear_map, py::const_), "M"_a,
             "CORALean's zonotope.linear_map, run by the oracle")
        .def("plus", py::overload_cast<const LZ &>(&LZ::plus, py::const_), "Z2"_a,
             "CORALean's zonotope.plus, run by the oracle")
        .def("plus", py::overload_cast<const LI &>(&LZ::plus, py::const_), "I"_a,
             "CORALean's zonotope.plus, run by the oracle")
        .def("linear_combination", &LZ::linear_combination, "Z2"_a,
             "CORALean's zonotope.linear_combination, run by the oracle")
        // sums and reductions in the oracle
        .def("exact_plus", &LZ::exact_plus, "Z2"_a, "k"_a,
             "CORALean's zonotope.exact_plus, run by the oracle")
        .def("reduce", &LZ::reduce, "order"_a, "method"_a = "girard",
             "CORALean's zonotope.reduce, run by the oracle")
        .def("abs", &LZ::abs, "CORALean's zonotope.abs, run by the oracle")
        .def("intersect", &LZ::intersect, "Z2"_a,
             "CORALean's zonotope.intersect, run by the oracle")
        .def("convex_hull", &LZ::convex_hull, "Z2"_a,
             "CORALean's zonotope.convex_hull, run by the oracle")
        .def("enclose", &LZ::enclose, "Z2"_a, "CORALean's zonotope.enclose, run by the oracle")
        .def("cart_prod", &LZ::cart_prod, "Z2"_a,
             "CORALean's zonotope.cart_prod, run by the oracle")
        .def("compact", &LZ::compact, "method"_a = "zeros", "tol"_a = 0.0,
             "CORALean's zonotope.compact, run by the oracle")
        .def("intersect_strip", &LZ::intersect_strip, "C"_a, "phi"_a, "y"_a, "Lambda"_a,
             "CORALean's zonotope.intersect_strip, run by the oracle")
        .def("quad_map", &LZ::quad_map, "Q"_a, "CORALean's zonotope.quad_map, run by the oracle")
        .def("cub_map", &LZ::cub_map, "T"_a, "CORALean's zonotope.cub_map, run by the oracle")
        .def_static("enclose_many", &LZ::enclose_many, "Zs"_a, "direction"_a, "p"_a = 20,
                    "CORALean's zonotope.enclose_many, run by the oracle")
        // enclosures of points and sets in the oracle
        .def_static("enclose_points", &LZ::enclose_points, "points"_a, "U"_a,
                    "method"_a = "stursberg",
                    "CORALean's zonotope.enclose_points, run by the oracle")
        .def_static("filter_out", &LZ::filter_out, "Zs"_a,
                    "CORALean's zonotope.filter_out, run by the oracle")
        .def_static("origin", &LZ::origin, "n"_a, "CORALean's zonotope.origin, run by the oracle")
        .def("project", &LZ::project, "dims"_a, "CORALean's zonotope.project, run by the oracle")
        .def("lift", &LZ::lift, "N"_a, "dims"_a, "CORALean's zonotope.lift, run by the oracle")
        .def("project_high_dim", &LZ::project_high_dim, "N"_a, "dims"_a,
             "CORALean's zonotope.project_high_dim, run by the oracle")
        .def("rotate", &LZ::rotate, "p"_a, "i"_a, "j"_a, "angle"_a,
             "CORALean's zonotope.rotate, run by the oracle")
        .def("split", &LZ::split, "gen"_a, "CORALean's zonotope.split, run by the oracle")
        .def("split_first_gen", &LZ::split_first_gen,
             "CORALean's zonotope.split_first_gen, run by the oracle")
        .def("split_longest_gen", &LZ::split_longest_gen,
             "CORALean's zonotope.split_longest_gen, run by the oracle")
        .def("constr_sat", &LZ::constr_sat, "C"_a, "d"_a,
             "CORALean's zonotope.constr_sat, run by the oracle")
        // predicates of the oracle's zonotope
        .def("is_bounded", &LZ::is_bounded, "tol"_a = 0.0,
             "CORALean's zonotope.is_bounded, run by the oracle")
        .def("is_full_dim", &LZ::is_full_dim, "tol"_a = 0.0,
             "CORALean's zonotope.is_full_dim, run by the oracle")
        .def("isequal", &LZ::isequal, "Z2"_a, "tol"_a = 0.0,
             "CORALean's zonotope.isequal, run by the oracle")
        .def("representsa", &LZ::representsa, "type"_a, "tol"_a = 0.0,
             "CORALean's zonotope.representsa, run by the oracle")
        .def("contains", &LZ::contains, "Z2"_a, "tol"_a = 0.0,
             "CORALean's zonotope.contains, run by the oracle")
        .def("is_intersecting", &LZ::is_intersecting, "Z2"_a, "tol"_a = 0.0,
             "CORALean's zonotope.is_intersecting, run by the oracle")
        .def("norm", &LZ::norm, "type"_a = 2, "mode"_a = "ub",
             "CORALean's zonotope.norm, run by the oracle")
        .def("support_function", &LZ::support_function, "dir"_a, "type"_a = "upper",
             "CORALean's zonotope.support_function, run by the oracle")
        .def("radius", &LZ::radius, "k"_a = 20, "CORALean's zonotope.radius, run by the oracle")
        .def("minnorm", &LZ::minnorm, "p"_a = 20, "CORALean's zonotope.minnorm, run by the oracle")
        .def("generator_length", &LZ::generator_length, "k"_a = 20,
             "CORALean's zonotope.generator_length, run by the oracle")
        // dominant directions in the oracle
        .def("dominant_directions", py::overload_cast<>(&LZ::dominant_directions, py::const_),
             "CORALean's zonotope.dominant_directions, run by the oracle")
        .def("dominant_directions",
             py::overload_cast<int, int>(&LZ::dominant_directions, py::const_), "filter_length1"_a,
             "filter_length2"_a, "CORALean's zonotope.dominant_directions, run by the oracle")
        .def("vertices", &LZ::vertices, "CORALean's zonotope.vertices, run by the oracle")
        .def("interval", &LZ::interval, "CORALean's zonotope.interval, run by the oracle")
        .def(
            "gather", [](const LZ &Z) { return Z.gather(); },
            "(cora.Zonotope, cora.Interval): the nominal part and the error, exactly");

    py::class_<lean::Fullspace>(m, "Fullspace", "The whole space R^n and the oracle's operations")
        .def(py::init<int64_t>(), "n"_a)
        .def(py::init<const cora::Fullspace &>(), "F"_a, "The whole space of a cora.Fullspace")
        .def_property_readonly("dim", &lean::Fullspace::dim, "The number of dimensions n")
        .def_static("inf", &lean::Fullspace::inf, "n"_a,
                    "CORALean's fullspace.inf, run by the oracle")
        .def("box", &lean::Fullspace::box, "CORALean's fullspace.box, run by the oracle")
        .def("interval", &lean::Fullspace::interval,
             "CORALean's fullspace.interval, run by the oracle")
        // the oracle's full space
        .def("intersect",
             py::overload_cast<const lean::Fullspace &>(&lean::Fullspace::intersect, py::const_),
             "F2"_a, "CORALean's fullspace.intersect, run by the oracle",
             "CORALean's fullspace.intersect, run by the oracle")
        .def("intersect", py::overload_cast<const LI &>(&lean::Fullspace::intersect, py::const_),
             "I"_a, "CORALean's fullspace.intersect with a box, run by the oracle")
        .def("plus", &lean::Fullspace::plus, "F2"_a, "CORALean's fullspace.plus, run by the oracle")
        .def("convex_hull", &lean::Fullspace::convex_hull, "F2"_a,
             "CORALean's fullspace.convex_hull, run by the oracle")
        .def("contains", &lean::Fullspace::contains, "I"_a,
             "CORALean's fullspace.contains, run by the oracle")
        .def("is_intersecting", &lean::Fullspace::is_intersecting, "I"_a,
             "CORALean's fullspace.is_intersecting, run by the oracle")
        .def("isequal", &lean::Fullspace::isequal, "F2"_a, "tol"_a = 0.0,
             "CORALean's fullspace.isequal, run by the oracle")
        .def("is_bounded", &lean::Fullspace::is_bounded,
             "CORALean's fullspace.is_bounded, run by the oracle")
        .def("is_full_dim", &lean::Fullspace::is_full_dim,
             "CORALean's fullspace.is_full_dim, run by the oracle")
        .def("center", &lean::Fullspace::center, "CORALean's fullspace.center, run by the oracle")
        // enclosures and lifts of the oracle's full space
        .def_static("enclose_points", &lean::Fullspace::enclose_points, "points"_a,
                    "CORALean's fullspace.enclose_points, run by the oracle")
        .def("lift", &lean::Fullspace::lift, "N"_a, "dims"_a,
             "CORALean's fullspace.lift, run by the oracle")
        .def("project", &lean::Fullspace::project, "dims"_a,
             "CORALean's fullspace.project, run by the oracle")
        .def("project_high_dim", &lean::Fullspace::project_high_dim, "N"_a, "proj"_a,
             "CORALean's fullspace.project_high_dim, run by the oracle")
        .def("radius", &lean::Fullspace::radius, "CORALean's fullspace.radius, run by the oracle")
        .def("volume", &lean::Fullspace::volume, "CORALean's fullspace.volume, run by the oracle")
        .def("complement", &lean::Fullspace::complement,
             "CORALean's fullspace.complement, run by the oracle");

    py::class_<lean::ReachSet>(m, "ReachSet", "The reachable sets: lists of lean zonotopes")
        .def_readonly("time_interval", &lean::ReachSet::time_interval,
                      "The zonotopes that enclose the reachable set of each time interval")
        .def_readonly("time_point", &lean::ReachSet::time_point,
                      "The zonotopes that enclose the reachable set at each time point");

    py::class_<lean::LinearSys>(m, "LinearSys", "The linear system x' = A x, reached by CORALean")
        .def(py::init([](const Array &A) { return lean::LinearSys(host_tensor(A, true)); }),
             py::arg("A"))
        // reach in the oracle
        .def(
            "reach",
            [](const lean::LinearSys &sys, const lean::Zonotope &initial_set, double t_final,
               const py::kwargs &kwargs) {
                py::dict rest;
                for (const auto &item : kwargs) rest[item.first] = item.second;
                alg::Standard standard;
                standard.zonotope_order = 10;
                const alg::Standard a = fill(standard, rest);
                refuse_leftovers(rest, field_names<alg::Standard>(), "lean reach");
                return sys.reach(initial_set, t_final, a);
            },
            py::arg("initial_set"), py::arg("t_final"),
            "reach(initial_set=X0, t_final=T, time_step=dt, taylor_terms=10, zonotope_order=10)");
}

} // namespace


// ===========================================  MAIN  =========================================== //

PYBIND11_MODULE(_cora, m) {
    m.doc() = "CORA.cpp: set-based computing on Eigen (numpy), and on libtorch (torch) when built "
              "with it";

    py::class_<cora::Rng>(m, "Rng", "Random numbers; a seed makes a draw repeatable.")
        .def(py::init<uint64_t>(), py::arg("seed") = 0);
#ifdef CORACPP_PYTHON_TORCH
    m.attr("has_torch") = true;
#else
    m.attr("has_torch") = false;
    set_default_backend(
        "eigen"); // the Python package has no torch tensors, whatever the library has
#endif
    m.def("_warn", &warn, py::arg("identifier"), py::arg("message"),
          "Hands a warning to the hook (for the tests of the hook)");

    bind_errors(m);
    bind_settings(m);
    configure_threads(); // the pools, before torch or numpy start any work
    bind_cont_set(m);
    bind_interval(m);
    bind_empty_set(m);
    bind_fullspace(m);
    bind_zonotope(m);
    bind_con_zonotope(m);
    bind_zono_bundle(m);
    bind_polytope(m);
    bind_ellipsoid(m);
    bind_capsule(m);
    bind_poly_zonotope(m);
    bind_interval_matrix(m);
    bind_mat_zonotope(m);
    // the operators, the problems and the systems
    bind_operators(m);
    bind_problems(m);
    bind_linear_sys(m);
    bind_nonlinear_sys(m);
    bind_level_set(m);
    bind_taylm(m);
    bind_nonlin_das_sys(m);
    bind_lin_param_sys(m);
    bind_neural_network(m);
    bind_plot(m);
    bind_specification(m);
    bind_hybrid(m);
    // the oracle and the other modules
    bind_lean(m);
}

// NOLINTEND(backend,clones)

// ---------------------------------------  END OF CODE  ---------------------------------------- //
