// bindings - the compiled half of the `cora` Python package (`_cora`)
//
// The classes are the C++ classes, with the same names, so code reads the same in both
// languages and a batch lives in the object, not in the call. Python gets properties for what a
// set is (Z.center, Z.generators, Z.dim), operators (+, @, *, &, |, in, unary -), and keyword
// arguments for a problem and its algorithm:
//
//     X0  = cora.Zonotope(c, G)                 # c (..., n), G (..., n, m): one set or a batch
//     sys = cora.LinearSys(A)                   # A (..., n, n): one system or a batch
//     R   = sys.reach(initial_set=X0, t_final=2.0, alg="standard", time_step=0.1)
//     R.time_interval[k].center
//
// A Python subclass of cora.ContSet implements _dim, _center, _support_function, ... and then
// works with the checked entry points, the binary registry (register_binary) and plotting; an
// algorithm registered from Python (cora.register_algorithm) gets the problem checks of the
// built-in ones. Errors arrive as cora.errors classes, warnings through warnings.warn.
//
// Syntax:   cora.Zonotope, Interval, ContSet, LinearSys, NonlinearSys, NeuralNetwork, Expr,
//           ReachSet, Specification, VerifyResult, Rng, backend, checks, plus, register_binary
// See also: convert.h, the C++ headers, which document every method

#include "global/python/convert.h"

#include "cora/cont_dynamics/linear_sys/linear_sys.h"
#include "cora/cont_dynamics/nonlinear_sys/nonlinear_sys.h"
#include "cora/cont_set/registry.h"
#include "cora/global/oracle/linear_sys.h"
#include "cora/global/oracle/oracle.h"
#include "cora/global/oracle/request.h"
#include "cora/global/plot/plot.h"
#include "cora/global/rng.h"
#include "cora/nn/neural_network/neural_network.h"
#include "cora/specification/specification.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

using namespace cora;
using namespace cora::python;

namespace {

// Errors, warnings and settings -------------------------------------------------------------------

/// cora::Error becomes the class of its identifier in cora.errors (a ValueError, ...).
void bind_errors(py::module_ &) {
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
    m.def(
        "_default_tol",
        [](const std::string &dtype) {
            return default_tol(dtype == "float32" ? DType::float32 : DType::float64);
        },
        py::arg("dtype"), "The tolerance of a dtype ('float64' or 'float32') without any override");
    m.def(
        "backend",
        [](const std::string &spec) {
#ifndef CORACPP_PYTHON_TORCH
            if (spec.rfind("torch", 0) == 0)
                CORA_FAIL(Error::install, "cora: this build of the Python package has no torch; "
                                          "use cora.backend('eigen')");
#endif
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

// Sets
// ---------------------------------------------------------------------------------------------

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
            const auto a =
                py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(p);
            Polygon polygon;
            for (py::ssize_t i = 0; i < a.shape(0); ++i)
                polygon.push_back({a.at(i, 0), a.at(i, 1)});
            out.push_back(polygon);
        }
        return out;
    }
    std::vector<std::vector<Point3>> vertices_3d_impl() const override {
        py::gil_scoped_acquire gil;
        std::vector<std::vector<Point3>> out;
        for (const py::handle &p : call("_vertices_3d")) {
            const auto a =
                py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(p);
            std::vector<Point3> corners;
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
    m.def(
        "is_intersecting",
        [](const py::object &S1, const py::object &S2, Tol tol) {
            return is_intersecting(set_of(S1), set_of(S2), tol);
        },
        py::arg("S1"), py::arg("S2"), py::arg("tol") = py::none(),
        "Whether S1 and S2 intersect, up to tol (None: the repository tolerance)");
    m.def(
        "binary_pairs",
        [] {
            py::list out;
            for (const BinaryPair &p : binary_pairs())
                out.append(py::make_tuple(binary_op_name(p.op), p.left, p.right));
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
                    o->release();  // the interpreter is gone (static destruction): leak it
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
            if (o == BinaryOp::contains || o == BinaryOp::is_intersecting) {
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

/// What every set has; bound on the base class, so any set works wherever a set is expected,
/// including the operators between sets of any classes (through the registry).
void bind_cont_set(py::module_ &m) {
    py::class_<ContSet, PyContSet>(m, "ContSet",
                                   "The abstract set; a Python subclass implements _dim, _center, "
                                   "_support_function, _interval, _rand_point, _project, "
                                   "_vertices, _vertices_3d")
        .def(py::init<>())
        .def_property_readonly("dim", &ContSet::dim, "The dimension n")
        .def_property_readonly(
            "center", [](const ContSet &S) { return bounds(S.center()); }, "The center (..., n)")
        .def_property_readonly("class_name", &ContSet::class_name)
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
                py::list corners;
                for (const std::vector<Point3> &p : S.vertices_3d())
                    corners.append(points3_array(p));
                return corners;
            },
            "The vertices of a three-dimensional set, one (k, 3) array per batch member")
        .def("display", &ContSet::display)
        .def("__repr__", &ContSet::display)
        .def("__str__", &ContSet::display)
        .def("__add__",
             [](const py::object &S1, const py::object &S2) -> py::object {
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
            CORA_FAIL(Error::not_supported,
                      "point containment is implemented for intervals and zonotopes only");
        });
    bind_registry(m);
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

void bind_interval(py::module_ &m) {
    py::class_<Interval, ContSet>(m, "Interval",
                                  "A box, or a matrix of intervals: inf <= x <= sup.")
        .def(py::init([](const py::object &inf, const py::object &sup) {
                 return Interval(vector_of(inf, "inf"), vector_of(sup, "sup"));
             }),
             py::arg("inf"), py::arg("sup"), "The box of the vectors inf <= sup (..., n)")
        .def_static(
            "matrix",
            [](const py::object &inf, const py::object &sup) {
                return Interval(tensor_of(inf, "inf"), tensor_of(sup, "sup"));
            },
            py::arg("inf"), py::arg("sup"), "An interval matrix: bounds (..., n, k)")
        .def_static("stack", &Interval::stack, py::arg("intervals"),
                    "A batch of intervals as one; the batch is the leading dimension")
        .def_static("generate_random", &Interval::generate_random, py::arg("n"), py::arg("rng"))
        .def_property_readonly("inf", [](const Interval &I) { return bounds(I.inf()); })
        .def_property_readonly("sup", [](const Interval &I) { return bounds(I.sup()); })
        .def_property_readonly("rad", [](const Interval &I) { return bounds(I.rad()); })
        .def(
            "linear_map",
            [](const Interval &I, const py::object &M) {
                return I.linear_map(tensor_of(M, "the matrix"));
            },
            py::arg("M"), "M I enclosed by its interval hull")
        .def("plus", &Interval::plus, py::arg("I2"))
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
        .def("enclose", &Interval::enclose, py::arg("I2"), "A box around two boxes")
        .def_static(
            "enclose_points",
            [](const py::object &points) {
                return Interval::enclose_points(tensor_of(points, "the points"));
            },
            py::arg("points"), "The smallest box around the points, the columns of (n, N)")
        .def_static("empty", &Interval::empty, py::arg("n"), "The empty interval of n dimensions")
        .def_static(
            "horzcat",
            [](const py::list &items) {
                std::vector<Interval> Is;
                for (const py::handle &item : items) {
                    if (py::isinstance<Interval>(item)) {
                        Is.push_back(py::cast<Interval>(item));
                    } else {
                        const Tensor p = vector_of(item, "the number");
                        Is.emplace_back(p, p);
                    }
                }
                return Interval::horzcat(Is);
            },
            py::arg("items"), "Intervals (or points, as arrays) joined side by side")
        .def(
            "enlarge",
            [](const Interval &I, const py::object &factor) {
                if (is_number(factor)) return I.enlarge(py::cast<double>(factor));
                return I.enlarge(vector_of(factor, "the factor"));
            },
            py::arg("factor"), "The box scaled about its center by a number or per dimension")
        .def("ctranspose", &Interval::ctranspose, "The transpose of an interval matrix")
        .def("diag", &Interval::diag, py::arg("k") = 0,
             "A vector as a diagonal matrix, a matrix as its k-th diagonal")
        .def("dirac", &Interval::dirac, py::arg("n") = 0,
             "The n-th derivative of the Dirac delta on the box")
        .def(
            "infimum", [](const Interval &I) { return bounds(I.infimum()); }, "The lower bound")
        .def("interval_matrix", &Interval::interval_matrix,
             "The interval matrix [c - r, c + r] of the center and the radius")
        .def("is_bounded", &Interval::is_bounded, "Whether no bound is infinite")
        .def("is_full_dim", &Interval::is_full_dim, "Whether the box has a nonempty interior")
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
                if (py::isinstance<Interval>(other)) return I.isequal(py::cast<Interval>(other), tol);
                return I.isequal(tensor_of(other, "the point"), tol);
            },
            py::arg("other"), py::arg("tol") = py::none(),
            "Whether the bounds agree up to tol; infinite bounds must be the same")
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
        .def("matrix_power", &Interval::matrix_power, py::arg("exponent"),
             "The power of a square interval matrix")
        .def(
            "max",
            [](const Interval &I, const py::object &other) -> py::object {
                if (other.is_none()) return bounds(I.max());
                if (py::isinstance<ContSet>(other)) return py::cast(I.max(py::cast<const ContSet &>(other)));
                return py::cast(I.max(operand(I, other, "the operand")));
            },
            py::arg("other") = py::none(),
            "The elementwise larger of two boxes; without an argument the upper bound")
        .def(
            "min",
            [](const Interval &I, const py::object &other) -> py::object {
                if (other.is_none()) return bounds(I.min());
                if (py::isinstance<ContSet>(other)) return py::cast(I.min(py::cast<const ContSet &>(other)));
                return py::cast(I.min(operand(I, other, "the operand")));
            },
            py::arg("other") = py::none(),
            "The elementwise smaller of two boxes; without an argument the lower bound")
        .def(
            "minkowski_difference",
            [](const Interval &I, const py::object &S, const std::optional<std::string> &type) {
                if (py::isinstance<Interval>(S)) return I.minkowski_difference(py::cast<Interval>(S));
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
                return I.unite(vector_of(other, "the point"));
            },
            py::arg("other"), "The smallest box around the box and a box or a point")
        .def(
            "norm",
            [](const Interval &I, const py::object &type) {
                return scalar(I.norm(norm_of(type)));
            },
            py::arg("type") = 2, "The norm of the largest member: 2 (default), 1, inf or 'fro'")
        .def("lift", &Interval::lift, py::arg("N"), py::arg("dims"),
             "The box in N dimensions at the coordinates dims (0-based), unbounded elsewhere")
        .def_static("origin", &Interval::origin, py::arg("n"), "The box {0} of n dimensions")
        .def("log", &Interval::log, "The image under the natural logarithm")
        .def(
            "contains",
            [](const Interval &I, const py::object &p, Tol tol) {
                return I.contains(vector_of(p, "the point"), tol);
            },
            py::arg("p"), py::arg("tol") = py::none(),
            "Whether p is in the box enlarged by tol per coordinate");
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
        .def(py::init([](const py::object &c, const py::object &G) {
                 return Zonotope(vector_of(c, "the center"), tensor_of(G, "the generators"));
             }),
             py::arg("center"), py::arg("generators"), "center (..., n) and generators (..., n, m)")
        .def_static("stack", &Zonotope::stack, py::arg("zonotopes"),
                    "A batch of zonotopes as one; fewer generators are padded with zeros")
        .def_static("generate_random", &Zonotope::generate_random, py::arg("n"), py::arg("m"),
                    py::arg("rng"))
        .def_property_readonly("center", [](const Zonotope &Z) { return vector(Z.center()); })
        .def_property_readonly("generators",
                               [](const Zonotope &Z) { return matrix(Z.generators()); })
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
             py::arg("Z2"))
        .def("plus", py::overload_cast<const Interval &>(&Zonotope::plus, py::const_), py::arg("I"))
        .def("linear_combination", &Zonotope::linear_combination, py::arg("Z2"))
        .def("reduce", &Zonotope::reduce, py::arg("order"),
             "At most order * n generators; the smallest are enclosed by a box (Girard)")
        .def("abs", &Zonotope::abs, "|c| and |G| entrywise (not the image of the set)")
        .def("box", &Zonotope::box, "The box around the zonotope, as a zonotope")
        .def("compact", &Zonotope::compact, py::arg("method") = "zeros",
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
        .def("enclose", &Zonotope::enclose, py::arg("Z2"), "A zonotope that encloses both")
        .def(
            "convex_hull",
            [](const Zonotope &Z, const py::object &S) {
                if (py::isinstance<Zonotope>(S)) return Z.convex_hull(py::cast<Zonotope>(S));
                if (py::isinstance<Interval>(S)) return Z.convex_hull(py::cast<Interval>(S));
                return Z.convex_hull(vector_of(S, "the point"));
            },
            py::arg("S"), "A zonotope that encloses the convex hull with a zonotope, box or point")
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
        .def(
            "dominant_directions",
            [](const Zonotope &Z, const std::optional<int> &filter_length1,
               const std::optional<int> &filter_length2) {
                return matrix(Z.dominant_directions(filter_length1, filter_length2));
            },
            py::arg("filter_length1") = py::none(), py::arg("filter_length2") = py::none(),
            "n generators (columns) that span a tight parallelotope around the zonotope")
        .def(
            "contains",
            [](const Zonotope &Z, const py::object &S, const std::string &method, Tol tol) {
                if (py::isinstance<Zonotope>(S)) return Z.contains(py::cast<Zonotope>(S), method, tol);
                if (py::isinstance<Interval>(S)) return Z.contains(py::cast<Interval>(S), method, tol);
                return Z.contains(tensor_of(S, "the points"), method, tol);
            },
            py::arg("S"), py::arg("method") = "exact",
            py::arg("tol") = py::none(),
            "Whether the zonotope contains the point(s) (n,) or (n, N), a zonotope or a box")
        .def(
            "boundary_point",
            [](const Zonotope &Z, const py::object &dir, const std::optional<py::object> &start) {
                return vector(start ? Z.boundary_point(vector_of(dir, "dir"), vector_of(*start, "start"))
                                    : Z.boundary_point(vector_of(dir, "dir")));
            },
            py::arg("dir"), py::arg("start") = py::none(),
            "Where the ray from start (default: the center) in direction dir leaves the zonotope")
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
        .def("is_full_dim", &Zonotope::is_full_dim, py::arg("tol") = 1e-6,
             "Whether the generators span the space (singular values up to tol count as zero)")
        .def(
            "is_full_dim_subspace",
            [](const Zonotope &Z, double tol) {
                const auto [res, subspace] = Z.is_full_dim_subspace(tol);
                return py::make_tuple(res, matrix(subspace));
            },
            py::arg("tol") = 1e-6, "(res, subspace): is_full_dim and a basis (n, r) of the span")
        .def(
            "isequal",
            [](const Zonotope &Z, const py::object &S, double tol) {
                if (py::isinstance<Zonotope>(S)) return Z.isequal(py::cast<Zonotope>(S), tol);
                if (py::isinstance<Interval>(S)) return Z.isequal(py::cast<Interval>(S), tol);
                return Z.isequal(tensor_of(S, "the point"), tol);
            },
            py::arg("S"), py::arg("tol") = std::numeric_limits<double>::epsilon(),
            "Whether S (a zonotope, a box or a point) is the same set, up to tol")
        .def(
            "is_intersecting",
            [](const Zonotope &Z, const py::object &S, const std::string &type, double tol) {
                if (py::isinstance<Zonotope>(S))
                    return Z.is_intersecting(py::cast<Zonotope>(S), type, tol);
                if (py::isinstance<Interval>(S))
                    return Z.is_intersecting(py::cast<Interval>(S), type, tol);
                return Z.is_intersecting(tensor_of(S, "the points"), type, tol);
            },
            py::arg("S"), py::arg("type") = "exact", py::arg("tol") = 1e-8,
            "Whether Z and a zonotope, a box or points intersect; 'exact' (default) or 'approx'")
        .def("lift", &Zonotope::lift, py::arg("N"), py::arg("dims"),
             "In N dimensions, with state i at position dims[i] (a permutation)")
        .def(
            "norm",
            [](const Zonotope &Z, double type, const std::string &mode) {
                return scalar(Z.norm(type, mode));
            },
            py::arg("type") = 2.0, py::arg("mode") = "ub",
            "An upper bound of the 1-, 2- or inf-norm of the points (mode 'ub')")
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
                const auto [Z, inverse] = Zonotope::enclose_many(Zs, vector_of(direction, "direction"));
                return py::make_tuple(Z, matrix(inverse));
            },
            py::arg("zonotopes"), py::arg("direction"),
            "(Z, rot_inv): a box oriented along the direction around the zonotopes, and the "
            "inverse of its frame")
        .def(
            "mink_diff",
            [](const Zonotope &Z, const py::object &S, const std::string &method) {
                if (py::isinstance<Zonotope>(S)) return Z.mink_diff(py::cast<Zonotope>(S), method);
                if (py::isinstance<Interval>(S)) return Z.mink_diff(py::cast<Interval>(S), method);
                return Z.mink_diff(vector_of(S, "the point"), method);
            },
            py::arg("S"), py::arg("method") = "approx",
            "The Minkowski difference with a zonotope, a box or a point; methods: approx, inner, "
            "exact, outer, outer:coarse");
}

/// The operators with numbers and arrays, as CORA writes them: s * S, M * S and M @ S (the linear
/// map), S + v, S - v, -S. What an operator does not understand it hands back to Python
/// (NotImplemented), which raises the TypeError.
void bind_operators(py::module_ &m) {
    const py::object Z = m.attr("Zonotope"), I = m.attr("Interval");
    Z.attr("__array_ufunc__") = py::none(); // numpy defers to the operators below
    I.attr("__array_ufunc__") = py::none();
    const auto method = [](const py::object &cls, const char *name, auto f) {
        cls.attr(name) = py::cpp_function(f, py::is_method(cls));
    };

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
    method(I, "__le__", [](const Interval &A, const py::object &o) -> py::object {
        if (py::isinstance<Interval>(o)) return py::cast(A <= py::cast<Interval>(o));
        return not_implemented();
    });
    method(I, "__lt__", [](const Interval &A, const py::object &o) -> py::object {
        if (py::isinstance<Interval>(o)) return py::cast(A < py::cast<Interval>(o));
        return not_implemented();
    });
    method(I, "__neg__", [](const Interval &A) { return -A; });
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
}

// Dynamics ---------------------------------------------------------------------------------------

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
        known.push_back("alg");
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
    py::class_<ReachSet>(m, "ReachSet", "The reachable sets: lists of zonotopes.")
        .def(py::init([](std::vector<Zonotope> time_interval, std::vector<Zonotope> time_point,
                         double time_step) {
                 return ReachSet{std::move(time_interval), std::move(time_point), time_step};
             }),
             py::arg("time_interval") = std::vector<Zonotope>(),
             py::arg("time_point") = std::vector<Zonotope>(), py::arg("time_step") = 0.0)
        .def_readonly("time_step", &ReachSet::time_step, "the step between the sets; 0 if unknown")
        .def_readonly("time_interval", &ReachSet::time_interval,
                      "time_interval[k] over [k*time_step, (k+1)*time_step]")
        .def_readonly("time_point", &ReachSet::time_point, "time_point[k] at k*time_step");

    py::class_<ReachProblem>(m, "ReachProblem", "What reach and verify are asked (CORA's params).")
        .def_readonly("initial_set", &ReachProblem::initial_set)
        .def_readonly("input_set", &ReachProblem::input_set)
        .def_readonly("t_final", &ReachProblem::t_final);

    py::class_<Falsification>(m, "Falsification", "A trajectory that violates a specification.")
        .def_property_readonly(
            "x0", [](const Falsification &f) { return vector(f.x0); }, "the state at time 0")
        .def_readonly("time", &Falsification::time, "the time it is violated");

    py::class_<VerifyResult>(m, "VerifyResult", "The outcome of verify.")
        .def_readonly("verified", &VerifyResult::verified)
        .def_readonly("computation_time", &VerifyResult::computation_time, "seconds spent")
        .def_readonly("iterations", &VerifyResult::iterations, "passes of the adaptive loop")
        .def_readonly("time_step", &VerifyResult::time_step, "step size of the last pass")
        .def_readonly("num_steps", &VerifyResult::num_steps, "number of steps of the last pass")
        .def_readonly("falsification", &VerifyResult::falsification, "a Falsification, or None");
}

void bind_linear_sys(py::module_ &m) {
    py::class_<LinearSys>(m, "LinearSys", "The linear system x' = A x + B u, y = C x.")
        .def(py::init([](const py::object &A, const py::object &B, const py::object &C,
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
                 return LinearSys(t(A, "A"), t(B, "B"),
                                  C.is_none() ? std::nullopt : std::optional<Tensor>(t(C, "C")));
             }),
             py::arg("A"), py::arg("B") = py::none(), py::arg("C") = py::none(),
             py::arg("custom_backward") = false,
             "x' = A x (+ B u), y = C x: A (..., n, n), B (n, m), C (p, n); without C the output "
             "is x")
        .def_property_readonly("A", [](const LinearSys &sys) { return matrix(sys.A()); })
        .def_property_readonly("B",
                               [](const LinearSys &sys) -> py::object {
                                   return sys.B() ? matrix(*sys.B()) : py::none();
                               })
        .def_property_readonly("C",
                               [](const LinearSys &sys) -> py::object {
                                   return sys.C() ? matrix(*sys.C()) : py::none();
                               })
        .def_property_readonly("dim", &LinearSys::dim)
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
        .def(
            "simulate_random",
            [](const py::object &self, const py::args &args, const py::kwargs &kwargs) {
                const LinearSys &sys = py::cast<const LinearSys &>(self);
                const int64_t points =
                    py::cast<int64_t>(required(kwargs, "points", "simulate_random"));
                cora::Rng &rng = py::cast<cora::Rng &>(required(kwargs, "rng", "simulate_random"));
                return run_with_kwargs<RandomSimulationProblem, alg::LinearSimulate>(
                    self, "simulate_random", args, kwargs, {"points", "rng"},
                    [&](const RandomSimulationProblem &p, const alg::LinearSimulate &a) {
                        return stack_time(sys.simulate_random(p, a, points, rng));
                    });
            },
            "simulate_random(initial_set=X0, t_final=T, time_step=dt, points=N, rng=rng, "
            "input_set=U): simulate from N random points of X0 (and constant inputs of U)")
        .def("correction_matrix_state", &LinearSys::correction_matrix_state, py::arg("time_step"),
             py::arg("taylor_terms"), "The interval matrix F of the curvature enlargement")
        .def("correction_matrix_input", &LinearSys::correction_matrix_input, py::arg("time_step"),
             py::arg("taylor_terms"), "The interval matrix G of the curvature of the input")
        .def("expm_remainder", &LinearSys::expm_remainder, py::arg("time_step"),
             py::arg("taylor_terms"), "The error box of the truncated series of e^{|A| dt}")
        .def("validate", &LinearSys::validate, py::arg("problem"),
             "The checks of a reach problem against the system")
        .def(
            "verify",
            [](const py::object &self, const py::args &args, const py::kwargs &kwargs) {
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
            "alg='reach_avoid_support_function' or 'reach_avoid_zonotope'): whether every "
            "specification holds for the outputs up to t_final");
}

/// The dynamics as symbolic expressions: `x` is a list of Expr, `f(x)` a list of Expr or numbers.
void bind_nonlinear_sys(py::module_ &m) {
    py::class_<Expr>(m, "Expr", "A symbolic expression of the states, differentiated by the system.")
        .def(py::init<double>(), py::arg("value"))
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
        .def("__pow__", [](const Expr &a, int n) { return pow(a, n); }, py::arg("n"))
        .def("diff", &Expr::diff, py::arg("i"), "The derivative with respect to the i-th state")
        .def("eval", py::overload_cast<const std::vector<double> &>(&Expr::eval, py::const_), py::arg("x"),
             "The value at the point x (a list of numbers)");
    py::implicitly_convertible<double, Expr>();
    py::implicitly_convertible<int, Expr>();
    m.def("sin", [](const Expr &a) { return sin(a); }, py::arg("e"));
    m.def("cos", [](const Expr &a) { return cos(a); }, py::arg("e"));
    m.def("exp", [](const Expr &a) { return exp(a); }, py::arg("e"));

    py::class_<NonlinearSys>(m, "NonlinearSys", "The nonlinear system x' = f(x).")
        .def(py::init([](const py::function &f, int64_t dim) {
                 // f runs once, here, on symbolic states.
                 return NonlinearSys(
                     [&f](const std::vector<Expr> &x) { return f(x).cast<std::vector<Expr>>(); }, dim);
             }),
             py::arg("f"), py::arg("dim"), "f(x) returns the dim components of the right-hand side")
        .def_property_readonly("dim", &NonlinearSys::dim)
        .def(
            "reach",
            [](const py::object &self, const py::args &args, const py::kwargs &kwargs) {
                const NonlinearSys &sys = py::cast<const NonlinearSys &>(self);
                return run_with_kwargs<ReachProblem, alg::NonlinearReach>(
                    self, "reach", args, kwargs, {},
                    [&](const ReachProblem &p, const alg::NonlinearReach &a) {
                        return py::cast(sys.reach(p, a));
                    });
            },
            "reach(initial_set=X0, t_final=T, alg='linearization', time_step=dt, taylor_terms=4, "
            "zonotope_order=50): the reachable sets")
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

void bind_neural_network(py::module_ &m) {
    py::class_<NeuralNetwork>(m, "NeuralNetwork",
                              "Affine layers with a ReLU after each but the last.")
        .def(py::init([](const std::vector<std::pair<py::object, py::object>> &layers) {
                 std::vector<NeuralNetwork::Layer> out;
                 for (const auto &[W, b] : layers)
                     out.push_back({tensor_of(W, "the weights"), tensor_of(b, "the bias")});
                 return NeuralNetwork(out);
             }),
             py::arg("layers"), "the layers as (W (out, in), b (out,)) pairs, first to last")
        .def_property_readonly("input_dim", &NeuralNetwork::input_dim)
        .def_property_readonly("output_dim", &NeuralNetwork::output_dim)
        .def(
            "evaluate",
            [](const NeuralNetwork &nn, const py::object &x) -> py::object {
                if (py::isinstance<Zonotope>(x))
                    return py::cast(nn.evaluate(py::cast<Zonotope>(x)));
                const Tensor y = nn.evaluate(tensor_of(x, "the points"));
                const bool one = py::hasattr(x, "ndim") && py::cast<int>(x.attr("ndim")) == 1;
                return one ? vector(y) : matrix(y);
            },
            py::arg("x"),
            "The outputs at a point (in,) or the columns of (in, N); for a zonotope, "
            "a zonotope that contains the outputs of every point");
}

// Plotting ------------------------------------------------------------------------------------
// The plotting logic is the C++ one (plot/plot.h): a Figure collects layers with every choice
// made, and the Python plot draws them with matplotlib.

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
        "label",      "color",  "face_color", "edge_color", "face_alpha", "line_width",
        "unify",      "filled", "set",        "time_points", "step",       "num_colors",
        "cidx"};
    for (const auto &item : kwargs) {
        const std::string key = py::cast<std::string>(item.first);
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
    const std::vector<std::string> known = {"standalone",     "coramacs",      "thin",  "thin_tolerance",
                                            "external_data", "width",         "height"};
    for (const auto &item : kwargs) {
        const std::string key = py::cast<std::string>(item.first);
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

py::tuple rgb(const Color &c) { return py::make_tuple(c.r, c.g, c.b); }

/// A layer of a figure as a dict, for the code that draws it.
py::dict layer_dict(const Figure::Layer &l) {
    py::dict d;
    switch (l.kind) {
    case Figure::Kind::polygons: d["kind"] = "polygons"; break;
    case Figure::Kind::polyline: d["kind"] = "polyline"; break;
    case Figure::Kind::points: d["kind"] = "points"; break;
    case Figure::Kind::region: d["kind"] = "region"; break;
    case Figure::Kind::surface: d["kind"] = "surface"; break;
    }
    py::list polygons, faces;
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
    py::class_<Figure>(m, "Figure", "The layers that plot puts into one axes, with every choice made.")
        .def(py::init<>())
        .def_readwrite("title", &Figure::title)
        .def_readwrite("xlabel", &Figure::xlabel)
        .def_readwrite("ylabel", &Figure::ylabel)
        .def_readwrite("zlabel", &Figure::zlabel)
        .def_readwrite("equal_axes", &Figure::equal_axes)
        .def_readwrite("azimuth", &Figure::azimuth)
        .def_readwrite("elevation", &Figure::elevation)
        .def_readwrite("xlim", &Figure::xlim)
        .def_readwrite("ylim", &Figure::ylim)
        .def_readwrite("zlim", &Figure::zlim)
        .def_readonly("dimension", &Figure::dimension, "2, or 3 once a surface is drawn")
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
            py::arg("obj"), py::arg("dims"))
        .def(
            "plot",
            [](Figure &fig, const Specification &spec, const std::vector<int64_t> &dims,
               const py::kwargs &kwargs) { plot(fig, spec, dims, options_of(kwargs)); },
            py::arg("obj"), py::arg("dims"))
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
                plot_over_time(fig, trajectories(x), dim, time_step, options_of(kwargs));
            },
            py::arg("obj"), py::arg("dim"), py::arg("time_step"),
            "Draws a dimension of simulations (steps, n, N) over time")
        .def(
            "plot_rand_point",
            [](Figure &fig, const ContSet &S, const std::vector<int64_t> &dims, int64_t N,
               Rng &rng, const py::kwargs &kwargs) {
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
            [](const Figure &fig, const py::kwargs &kwargs) { return fig.tikz(tikz_options_of(kwargs)); },
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
                return polygon_array(Figure::clip_region(corners, a, b, sign));
            },
            py::arg("polygon"), py::arg("a"), py::arg("b"), py::arg("sign"),
            "The part of a polygon where sign * (a.x - b) >= 0");

    py::class_<FigureGrid>(m, "FigureGrid", "Figures in rows and columns: SVG cells or a groupplot.")
        .def(py::init<int, int>(), py::arg("rows"), py::arg("cols"))
        .def_readwrite("share_x", &FigureGrid::share_x)
        .def_readwrite("share_y", &FigureGrid::share_y)
        .def_readwrite("width", &FigureGrid::width)
        .def_readwrite("height", &FigureGrid::height)
        .def_property_readonly("rows", &FigureGrid::rows)
        .def_property_readonly("cols", &FigureGrid::cols)
        .def(
            "at",
            [](FigureGrid &g, int row, int col) -> Figure & { return g.at(row, col); },
            py::arg("row"), py::arg("col"), py::return_value_policy::reference_internal,
            "The figure in a cell")
        .def(
            "set", [](FigureGrid &g, int row, int col, const Figure &f) { g.at(row, col) = f; },
            py::arg("row"), py::arg("col"), py::arg("figure"), "Puts a copy of a figure into a cell")
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

void bind_specification(py::module_ &m) {
    py::class_<Specification>(m, "Specification",
                              "Halfspaces {x | a.x <= b} a set must stay in (safe_set) or must not "
                              "touch (unsafe_set).")
        .def_static(
            "safe_set",
            [](const py::object &a, double b) {
                return Specification::safe_set(vector_of(a, "the normal"), b);
            },
            py::arg("a"), py::arg("b"))
        .def_static(
            "unsafe_set",
            [](const py::object &a, const py::object &b) {
                if (is_number(b))
                    return Specification::unsafe_set(vector_of(a, "the normal"),
                                                     py::cast<double>(b));
                // the rows of A with b: a polytope the sets must not touch
                const Tensor A = host_tensor(a, false);
                const std::vector<double> bs = py::cast<std::vector<double>>(b);
                std::vector<Halfspace> halfspaces;
                for (int64_t i = 0; i < A.size(-2); ++i)
                    halfspaces.push_back({A.slice(-2, i, i + 1).matrix_transpose(), bs[i]});
                return Specification::unsafe_set(std::move(halfspaces));
            },
            py::arg("a"), py::arg("b"),
            "One halfspace a.x <= b, or a polytope A x <= b (A a matrix, b a vector)")
        .def_property_readonly("type",
                               [](const Specification &spec) {
                                   return spec.type() == SpecType::safe_set ? "safe_set"
                                                                            : "unsafe_set";
                               })
        .def_property_readonly("halfspaces",
                               [](const Specification &spec) {
                                   // The halfspaces as host numpy arrays, whatever the backend.
                                   py::list out;
                                   for (const Halfspace &h : spec.halfspaces()) {
                                       const std::vector<double> a = h.a.data();
                                       out.append(py::make_tuple(
                                           py::array_t<double>(a.size(), a.data()), h.b));
                                   }
                                   return out;
                               })
        .def(
            "holds", [](const Specification &spec, const ContSet &S) { return spec.holds(S); },
            py::arg("S"), "One flag per batch element of the set")
        .def(
            "check", [](const Specification &spec, const ContSet &S) { return spec.check(S); },
            py::arg("S"), "Whether every batch element of the set satisfies the specification")
        .def(
            "check",
            [](const Specification &spec, const std::vector<Zonotope> &sets) {
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
    py::module_ m = top.def_submodule("lean", "Sound float sets computed by CORALean (needs CORACPP_ORACLE)");
    bind_scope<lean::DTypeScope, std::string>(m, "_DTypeScope", "A dtype in a with block");
    m.def(
        "dtype", [](const std::string &spec) { return new DType{spec, std::nullopt}; },
        py::arg("spec"),
        "with cora.lean.dtype('ieee:binary32'): new lean objects in that dtype in the block; "
        "'binary64', 'ieee:<format>', 'dyadic:<p>', 'fixedpoint:<f>', 'float', 'nearest'");
    m.def("set_default_dtype", &lean::set_default_dtype, py::arg("spec"),
          "The dtype of threads without a cora.lean.dtype block");
    m.def("current_dtype", &lean::dtype, "This thread's dtype");

    using namespace pybind11::literals;
    using LI = lean::Interval;
    using LZ = lean::Zonotope;
    using LT = lean::Tensor;
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
        .def_property_readonly("dtype", &LT::dtype);
    py::implicitly_convertible<py::array, LT>();
    py::implicitly_convertible<py::list, LT>();

    py::class_<lean::Certified>(m, "Certified", "An answer of the oracle, a proof if `proved`")
        .def_readonly("value", &lean::Certified::value)
        .def_readonly("proved", &lean::Certified::proved);
    py::class_<lean::Vertices>(m, "Vertices", "The vertices V (columns) and an error box E")
        .def_readonly("V", &lean::Vertices::V)
        .def_readonly("E", &lean::Vertices::E);

    py::class_<LI>(m, "Interval", "A box of lean values and the oracle's operations on it")
        .def(py::init<LT, LT>(), "inf"_a, "sup"_a)
        .def(py::init<const cora::Interval &>(), "I"_a, "The exact box of a cora.Interval")
        .def_static(
            "from_arrays",
            [](const py::object &inf, const py::object &sup) {
                return LI::from(host_tensor(inf, true), host_tensor(sup, true));
            },
            "inf"_a, "sup"_a)
        .def(
            "gather",
            [](const LI &I) {
                auto [inf, sup] = I.gather();
                return Interval(inf, sup);
            },
            "A cora.Interval; an error if a bound is not a double")
        .def_property_readonly("inf", &LI::inf)
        .def_property_readonly("sup", &LI::sup)
        .def_static("empty", &LI::empty, "n"_a)
        .def_static("origin", &LI::origin, "n"_a)
        .def_static("unbounded", &LI::unbounded, "n"_a)
        .def_static("enclose_points", &LI::enclose_points, "points"_a)
        .def_static("interval_matrix", &LI::interval_matrix, "c"_a, "r"_a)
        .def("abs", &LI::abs)
        .def("uminus", &LI::uminus)
        .def("uplus", &LI::uplus)
        .def("sign", &LI::sign)
        .def("round", &LI::round)
        .def("sqrt", &LI::sqrt, "p"_a = 20)
        .def("exp", &LI::exp, "p"_a = 20)
        .def("sin", &LI::sin, "p"_a = 20)
        .def("cos", &LI::cos, "p"_a = 20)
        .def("tan", &LI::tan, "p"_a = 20)
        .def("sinh", &LI::sinh, "p"_a = 20)
        .def("cosh", &LI::cosh, "p"_a = 20)
        .def("tanh", &LI::tanh, "p"_a = 20)
        .def("log", &LI::log, "p"_a = 20, "k"_a = 20)
        .def("asin", &LI::asin, "p"_a = 20, "k"_a = 20)
        .def("acos", &LI::acos, "p"_a = 20, "k"_a = 20)
        .def("atan", &LI::atan, "p"_a = 20, "k"_a = 20)
        .def("asinh", &LI::asinh, "p"_a = 20, "k"_a = 20)
        .def("acosh", &LI::acosh, "p"_a = 20, "k"_a = 20)
        .def("atanh", &LI::atanh, "p"_a = 20, "k"_a = 20)
        .def("atan2", &LI::atan2, "I2"_a, "p"_a = 20, "k"_a = 20)
        .def("enlarge", &LI::enlarge, "factor"_a)
        .def("power", &LI::power, "exponent"_a)
        .def("rdivide", &LI::rdivide, "I2"_a)
        .def("mrdivide", &LI::mrdivide, "d"_a)
        .def("plus", &LI::plus, "I2"_a)
        .def("minus", &LI::minus, "I2"_a)
        .def("times", &LI::times, "I2"_a)
        .def("unite", &LI::unite, "I2"_a)
        .def("convex_hull", &LI::convex_hull, "I2"_a)
        .def("mink_diff", &LI::mink_diff, "I2"_a)
        .def("min", &LI::min, "I2"_a)
        .def("max", &LI::max, "I2"_a)
        .def("intersect", &LI::intersect, "I2"_a)
        .def("enclose", &LI::enclose, "I2"_a)
        .def("cart_prod", &LI::cart_prod, "I2"_a)
        .def("linear_map", py::overload_cast<const LT &>(&LI::linear_map, py::const_), "M"_a)
        .def("linear_map", py::overload_cast<const LI &>(&LI::linear_map, py::const_), "M"_a)
        .def("quad_map", &LI::quad_map, "Q"_a)
        .def("le", &LI::le, "I2"_a, "tol"_a = 0.0)
        .def("lt", &LI::lt, "I2"_a, "tol"_a = 0.0)
        .def("isequal", &LI::isequal, "I2"_a, "tol"_a = 0.0)
        .def("is_full_dim", &LI::is_full_dim, "tol"_a = 0.0)
        .def("is_bounded", &LI::is_bounded, "tol"_a = 0.0)
        .def("representsa", &LI::representsa, "type"_a, "tol"_a = 0.0)
        .def("contains", py::overload_cast<const LI &, double>(&LI::contains, py::const_), "I2"_a,
             "tol"_a = 0.0)
        .def("contains", py::overload_cast<const LZ &, double>(&LI::contains, py::const_), "Z"_a,
             "tol"_a = 0.0)
        .def("is_intersecting", py::overload_cast<const LI &, double>(&LI::is_intersecting, py::const_),
             "I2"_a, "tol"_a = 0.0)
        .def("is_intersecting", py::overload_cast<const LZ &, double>(&LI::is_intersecting, py::const_),
             "Z"_a, "tol"_a = 0.0)
        .def("center", &LI::center)
        .def("rad", &LI::rad)
        .def("infimum", &LI::infimum)
        .def("supremum", &LI::supremum)
        .def("radius", &LI::radius, "type"_a = 2, "mode"_a = "ub")
        .def("norm", &LI::norm, "type"_a = 2, "mode"_a = "ub")
        .def("volume", &LI::volume)
        .def("support_function", &LI::support_function, "dir"_a, "type"_a = "upper")
        .def("project", &LI::project, "dims"_a)
        .def("lift", &LI::lift, "N"_a, "dims"_a)
        .def("project_high_dim", &LI::project_high_dim, "N"_a, "dims"_a)
        .def("split", &LI::split, "dim"_a)
        .def("partition", &LI::partition, "splits"_a, "ind"_a)
        .def("dirac", &LI::dirac, "n"_a)
        .def("boundary_point", &LI::boundary_point, "dir"_a, "start"_a)
        .def("transpose", &LI::transpose)
        .def("ctranspose", &LI::ctranspose)
        .def("prod", &LI::prod)
        .def("diag", &LI::diag)
        .def("kron", &LI::kron, "I2"_a)
        .def("horzcat", &LI::horzcat, "I2"_a)
        .def("vertcat", &LI::vertcat, "I2"_a)
        .def("sum", &LI::sum, "dim"_a)
        .def("cat", &LI::cat, "I2"_a, "dim"_a)
        .def("mpower", &LI::mpower, "exponent"_a)
        .def("tril", &LI::tril, "K"_a = 0)
        .def("triu", &LI::triu, "K"_a = 0)
        .def("reduce", &LI::reduce, "order"_a, "method"_a = "girard")
        .def("zonotope", &LI::zonotope)
        .def("vertices", &LI::vertices);

    py::class_<LZ>(m, "Zonotope", "A nominal zonotope plus an error box, and the oracle's operations")
        .def(py::init([](const py::object &c, const py::object &G) {
                 return LZ(host_tensor(c, true), host_tensor(G, true));
             }),
             "center"_a, "generators"_a, "An exact zonotope in the current dtype")
        .def(py::init<const Zonotope &>(), "Z"_a, "The exact zonotope of a cora.Zonotope")
        .def("nominal", &LZ::nominal, "c and G with an empty error box")
        .def("error", &LZ::error, "The error box")
        .def_property_readonly("center", &LZ::center)
        .def_property_readonly("generators", &LZ::generators)
        .def("enclosure", &LZ::enclosure, "nominal and error as one zonotope")
        .def("round_to", &LZ::round_to, "dtype"_a, "mode"_a = "nearest")
        .def("linear_map", py::overload_cast<const LT &>(&LZ::linear_map, py::const_), "M"_a)
        .def("linear_map", py::overload_cast<const LI &>(&LZ::linear_map, py::const_), "M"_a)
        .def("plus", py::overload_cast<const LZ &>(&LZ::plus, py::const_), "Z2"_a)
        .def("plus", py::overload_cast<const LI &>(&LZ::plus, py::const_), "I"_a)
        .def("linear_combination", &LZ::linear_combination, "Z2"_a)
        .def("exact_plus", &LZ::exact_plus, "Z2"_a, "k"_a)
        .def("reduce", &LZ::reduce, "order"_a, "method"_a = "girard")
        .def("abs", &LZ::abs)
        .def("intersect", &LZ::intersect, "Z2"_a)
        .def("convex_hull", &LZ::convex_hull, "Z2"_a)
        .def("enclose", &LZ::enclose, "Z2"_a)
        .def("cart_prod", &LZ::cart_prod, "Z2"_a)
        .def("compact", &LZ::compact, "method"_a = "zeros", "tol"_a = 0.0)
        .def("intersect_strip", &LZ::intersect_strip, "C"_a, "phi"_a, "y"_a, "Lambda"_a)
        .def("quad_map", &LZ::quad_map, "Q"_a)
        .def("cub_map", &LZ::cub_map, "T"_a)
        .def_static("enclose_many", &LZ::enclose_many, "Zs"_a, "direction"_a, "p"_a = 20)
        .def_static("enclose_points", &LZ::enclose_points, "points"_a, "U"_a,
                    "method"_a = "stursberg")
        .def_static("filter_out", &LZ::filter_out, "Zs"_a)
        .def_static("origin", &LZ::origin, "n"_a)
        .def("project", &LZ::project, "dims"_a)
        .def("lift", &LZ::lift, "N"_a, "dims"_a)
        .def("project_high_dim", &LZ::project_high_dim, "N"_a, "dims"_a)
        .def("rotate", &LZ::rotate, "p"_a, "i"_a, "j"_a, "angle"_a)
        .def("split", &LZ::split, "gen"_a)
        .def("split_first_gen", &LZ::split_first_gen)
        .def("split_longest_gen", &LZ::split_longest_gen)
        .def("constr_sat", &LZ::constr_sat, "C"_a, "d"_a)
        .def("is_bounded", &LZ::is_bounded, "tol"_a = 0.0)
        .def("is_full_dim", &LZ::is_full_dim, "tol"_a = 0.0)
        .def("isequal", &LZ::isequal, "Z2"_a, "tol"_a = 0.0)
        .def("representsa", &LZ::representsa, "type"_a, "tol"_a = 0.0)
        .def("contains", &LZ::contains, "Z2"_a, "tol"_a = 0.0)
        .def("is_intersecting", &LZ::is_intersecting, "Z2"_a, "tol"_a = 0.0)
        .def("norm", &LZ::norm, "type"_a = 2, "mode"_a = "ub")
        .def("support_function", &LZ::support_function, "dir"_a, "type"_a = "upper")
        .def("radius", &LZ::radius, "k"_a = 20)
        .def("minnorm", &LZ::minnorm, "p"_a = 20)
        .def("generator_length", &LZ::generator_length, "k"_a = 20)
        .def("dominant_directions",
             py::overload_cast<>(&LZ::dominant_directions, py::const_))
        .def("dominant_directions",
             py::overload_cast<int, int>(&LZ::dominant_directions, py::const_),
             "filter_length1"_a, "filter_length2"_a)
        .def("vertices", &LZ::vertices)
        .def("interval", &LZ::interval)
        .def(
            "gather", [](const LZ &Z) { return Z.gather(); },
            "(cora.Zonotope, cora.Interval): the nominal part and the error, exactly");

    py::class_<lean::ReachSet>(m, "ReachSet", "The reachable sets: lists of lean zonotopes")
        .def_readonly("time_interval", &lean::ReachSet::time_interval)
        .def_readonly("time_point", &lean::ReachSet::time_point);

    py::class_<lean::LinearSys>(m, "LinearSys", "The linear system x' = A x, reached by CORALean")
        .def(py::init([](const py::object &A) { return lean::LinearSys(host_tensor(A, true)); }),
             py::arg("A"))
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

PYBIND11_MODULE(_cora, m) {
    m.doc() = "CORA.cpp: set-based computing on Eigen (numpy), and on libtorch (torch) when built with it";

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
    bind_cont_set(m);
    bind_interval(m);
    bind_zonotope(m);
    bind_operators(m);
    bind_problems(m);
    bind_linear_sys(m);
    bind_nonlinear_sys(m);
    bind_neural_network(m);
    bind_plot(m);
    bind_specification(m);
    bind_lean(m);
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
