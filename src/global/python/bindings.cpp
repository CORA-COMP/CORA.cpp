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
        .def("__enter__", [](S &s) -> S & { s.scope.emplace(s.arg); return s; },
             py::return_value_policy::reference)
        .def("__exit__", [](S &s, const py::args &) { s.scope.reset(); });
}

void bind_settings(py::module_ &m) {
    using Checks = PyScope<ChecksScope, bool>;
    using Backend = PyScope<BackendScope, std::string>;
    bind_scope<ChecksScope, bool>(m, "_ChecksScope", "The checks on or off in a with block");
    bind_scope<BackendScope, std::string>(m, "_BackendScope", "A backend in a with block");
    m.def("checks", [](bool enabled) { return new Checks{enabled, std::nullopt}; },
          py::arg("enabled"), "with cora.checks(False): turns the input checks off in the block");
    m.def("set_checks_enabled", &set_checks_enabled, py::arg("enabled"),
          "The process default of the input checks (CORACPP_CHECKS=0 in the environment: off)");
    m.def("checks_enabled", &checks_enabled, "Whether this thread checks inputs");
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
}

// Sets ---------------------------------------------------------------------------------------------

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
            const auto a = py::array_t<double, py::array::c_style | py::array::forcecast>::ensure(p);
            Polygon polygon;
            for (py::ssize_t i = 0; i < a.shape(0); ++i) polygon.push_back({a.at(i, 0), a.at(i, 1)});
            out.push_back(polygon);
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
    m.def("contains",
          [](const py::object &S1, const py::object &S2, double tol) {
              return contains(set_of(S1), set_of(S2), tol);
          },
          py::arg("S1"), py::arg("S2"), py::arg("tol") = 100 * std::numeric_limits<double>::epsilon(),
          "Whether S2 is a subset of S1 enlarged by tol (infinity norm)");
    m.def("is_intersecting",
          [](const py::object &S1, const py::object &S2, double tol) {
              return is_intersecting(set_of(S1), set_of(S2), tol);
          },
          py::arg("S1"), py::arg("S2"), py::arg("tol") = 1e-8, "Whether S1 and S2 intersect, up to tol");
    m.def("binary_pairs",
          [] {
              py::list out;
              for (const BinaryPair &p : binary_pairs())
                  out.append(py::make_tuple(binary_op_name(p.op), p.left, p.right));
              return out;
          },
          "The pairs of classes each binary operation is implemented for: (op, left, right)");
    m.def(
        "register_binary",
        [](const std::string &op, const py::type &left, const py::type &right, const py::function &f) {
            // the function is kept alive, and freed with the GIL held
            auto fn = std::shared_ptr<py::object>(new py::object(f), [](py::object *o) {
                py::gil_scoped_acquire gil;
                delete o;
            });
            const BinaryOp o = binary_op(op);
            const std::string l = py::str(left.attr("__name__")), r = py::str(right.attr("__name__"));
            const auto ref = [](const ContSet &S) {
                return py::cast(&S, py::return_value_policy::reference);
            };
            if (o == BinaryOp::contains || o == BinaryOp::is_intersecting) {
                register_binary(o, l, r,
                                PredicateFunction([fn, ref](const ContSet &S1, const ContSet &S2, double tol) {
                                    py::gil_scoped_acquire gil;
                                    return py::cast<bool>((*fn)(ref(S1), ref(S2), tol));
                                }));
            } else {
                register_binary(o, l, r, SetFunction([fn, ref](const ContSet &S1, const ContSet &S2) {
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
                                   "_support_function, _interval, _rand_point, _project, _vertices")
        .def(py::init<>())
        .def_property_readonly("dim", &ContSet::dim, "The dimension n")
        .def_property_readonly("center", [](const ContSet &S) { return bounds(S.center()); },
                               "The center (..., n)")
        .def_property_readonly("class_name", &ContSet::class_name)
        .def("support_function",
             [](const ContSet &S, const py::object &d) {
                 return scalar(S.support_function(tensor_of(d, "the direction")));
             },
             py::arg("d"), "max of d'x over the set; d (..., n)")
        .def("interval", &ContSet::interval, "The interval hull")
        .def("rand_point",
             [](const ContSet &S, int64_t N, cora::Rng &rng) { return matrix(S.rand_point(N, rng)); },
             py::arg("N"), py::arg("rng"), "N random points as columns (..., n, N)")
        .def("project",
             [](const ContSet &S, const std::vector<int64_t> &dims) { return object_of(S.project(dims)); },
             py::arg("dims"), "The projection onto the dimensions dims (0-based)")
        .def("vertices",
             [](const ContSet &S) {
                 py::list polygons;
                 for (const Polygon &p : S.vertices()) polygons.append(polygon_array(p));
                 return polygons;
             },
             "The vertices of a two-dimensional set, one (k, 2) array per batch member")
        .def("display", &ContSet::display)
        .def("__repr__", &ContSet::display)
        .def("__str__", &ContSet::display)
        .def("__add__", [](const py::object &S1, const py::object &S2) -> py::object {
            if (!py::isinstance<ContSet>(S2)) return not_implemented();
            return binary<&plus>(S1, S2);
        })
        .def("__and__", [](const py::object &S1, const py::object &S2) -> py::object {
            if (!py::isinstance<ContSet>(S2)) return not_implemented();
            return binary<&intersect>(S1, S2);
        })
        .def("__or__", [](const py::object &S1, const py::object &S2) -> py::object {
            if (!py::isinstance<ContSet>(S2)) return not_implemented();
            return binary<&unite>(S1, S2);
        })
        .def("__contains__", [](const py::object &S1, const py::object &S2) {
            if (py::isinstance<ContSet>(S2)) return contains(set_of(S1), set_of(S2));
            if (py::isinstance<Interval>(S1))
                return py::cast<const Interval &>(S1).contains(tensor_of(S2, "the point"));
            CORA_FAIL(Error::not_supported, "point containment is implemented for intervals only");
        });
    bind_registry(m);
}

void bind_interval(py::module_ &m) {
    py::class_<Interval, ContSet>(m, "Interval", "A box, or a matrix of intervals: inf <= x <= sup.")
        .def(py::init([](const py::object &inf, const py::object &sup) {
                 return Interval(tensor_of(inf, "inf"), tensor_of(sup, "sup"));
             }),
             py::arg("inf"), py::arg("sup"), "The box of the vectors inf <= sup (..., n)")
        .def_static("matrix",
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
        .def("linear_map",
             [](const Interval &I, const py::object &M) { return I.linear_map(tensor_of(M, "the matrix")); },
             py::arg("M"), "M I enclosed by its interval hull")
        .def("plus", &Interval::plus, py::arg("I2"))
        .def("contains",
             [](const Interval &I, const py::object &p, double tol) {
                 return I.contains(tensor_of(p, "the point"), tol);
             },
             py::arg("p"), py::arg("tol") = 100 * std::numeric_limits<double>::epsilon(),
             "Whether p is in the box enlarged by tol per coordinate");
}

void bind_zonotope(py::module_ &m) {
    py::class_<Zonotope, ContSet>(m, "Zonotope", "The zonotope {c + G b : |b|_inf <= 1}.")
        .def(py::init([](const py::object &c, const py::object &G) {
                 return Zonotope(tensor_of(c, "the center"), tensor_of(G, "the generators"));
             }),
             py::arg("center"), py::arg("generators"), "center (..., n) and generators (..., n, m)")
        .def_static("stack", &Zonotope::stack, py::arg("zonotopes"),
                    "A batch of zonotopes as one; fewer generators are padded with zeros")
        .def_static("generate_random", &Zonotope::generate_random, py::arg("n"), py::arg("m"),
                    py::arg("rng"))
        .def_property_readonly("center", [](const Zonotope &Z) { return vector(Z.center()); })
        .def_property_readonly("generators", [](const Zonotope &Z) { return matrix(Z.generators()); })
        .def("rand_point",
             [](const Zonotope &Z, int64_t N, cora::Rng &rng, const std::string &type) {
                 return matrix(Z.rand_point(N, rng, type));
             },
             py::arg("N"), py::arg("rng"), py::arg("type") = "standard",
             "N random points as columns (..., n, N); type 'standard' or 'extreme'")
        .def("linear_map",
             [](const Zonotope &Z, const py::object &M) {
                 if (py::isinstance<Interval>(M)) return Z.linear_map(py::cast<Interval>(M));
                 return Z.linear_map(tensor_of(M, "the matrix"));
             },
             py::arg("M"), "M Z for a matrix (..., k, n), or an interval matrix")
        .def("plus", py::overload_cast<const Zonotope &>(&Zonotope::plus, py::const_), py::arg("Z2"))
        .def("plus", py::overload_cast<const Interval &>(&Zonotope::plus, py::const_), py::arg("I"))
        .def("linear_combination", &Zonotope::linear_combination, py::arg("Z2"))
        .def("reduce", &Zonotope::reduce, py::arg("order"),
             "At most order * n generators; the smallest are enclosed by a box (Girard)");
}

/// The operators with numbers and arrays, as CORA writes them: s * S, M * S and M @ S (the linear
/// map), S + v, S - v, -S. What an operator does not understand it hands back to Python
/// (NotImplemented), which raises the TypeError.
void bind_operators(py::module_ &m) {
    const py::object Z = m.attr("Zonotope"), I = m.attr("Interval");
    Z.attr("__array_ufunc__") = py::none();  // numpy defers to the operators below
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
        if (const std::optional<Tensor> v = as_tensor(o)) return py::cast(A - *v);
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
    for (const Tensor &x : points) all.append(py::cast(to_eigen(x)));
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
py::dict registered_algorithms() {
    return py::module_::import("cora.extend").attr("ALGORITHMS");
}

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
        CORA_FAIL(Error::wrong_value, std::string(operation) + ": the keyword '" + key + "' is missing");
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
    py::dict rest;  // a copy: the keys used are removed
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
        std::vector<std::string> known = field_names<P>();
        for (const std::string &f :
             algorithm_fields<V>(std::make_index_sequence<std::variant_size_v<V>>()))
            known.push_back(f);
        for (const std::string &k : extra) known.push_back(k);
        known.push_back("alg");
        refuse_leftovers(rest, known, where);
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
        .def(py::init([](std::vector<Zonotope> time_interval, std::vector<Zonotope> time_point) {
                 return ReachSet{std::move(time_interval), std::move(time_point)};
             }),
             py::arg("time_interval") = std::vector<Zonotope>(),
             py::arg("time_point") = std::vector<Zonotope>())
        .def_readonly("time_interval", &ReachSet::time_interval,
                      "time_interval[k] over [k*time_step, (k+1)*time_step]")
        .def_readonly("time_point", &ReachSet::time_point, "time_point[k] at k*time_step");

    py::class_<ReachProblem>(m, "ReachProblem", "What reach and verify are asked (CORA's params).")
        .def_readonly("initial_set", &ReachProblem::initial_set)
        .def_readonly("input_set", &ReachProblem::input_set)
        .def_readonly("t_final", &ReachProblem::t_final);

    py::class_<Falsification>(m, "Falsification", "A trajectory that violates a specification.")
        .def_property_readonly("x0", [](const Falsification &f) { return vector(f.x0); },
                               "the state at time 0")
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
             "x' = A x (+ B u), y = C x: A (..., n, n), B (n, m), C (p, n); without C the output is x")
        .def_property_readonly("A", [](const LinearSys &sys) { return matrix(sys.A()); })
        .def_property_readonly("B", [](const LinearSys &sys) -> py::object {
            return sys.B() ? matrix(*sys.B()) : py::none();
        })
        .def_property_readonly("C", [](const LinearSys &sys) -> py::object {
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
                const int64_t points = py::cast<int64_t>(required(kwargs, "points", "simulate_random"));
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
                const int64_t points = py::cast<int64_t>(required(kwargs, "points", "simulate_random"));
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
    py::class_<NeuralNetwork>(m, "NeuralNetwork", "Affine layers with a ReLU after each but the last.")
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
                if (py::isinstance<Zonotope>(x)) return py::cast(nn.evaluate(py::cast<Zonotope>(x)));
                const Tensor y = nn.evaluate(tensor_of(x, "the points"));
                const bool one = py::hasattr(x, "ndim") && py::cast<int>(x.attr("ndim")) == 1;
                return one ? vector(y) : matrix(y);
            },
            py::arg("x"), "The outputs at a point (in,) or the columns of (in, N); for a zonotope, "
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
    const std::vector<std::string> known = {"label",  "color",       "facecolor", "face_alpha",
                                            "line_width", "unify",   "filled",    "time_points",
                                            "step",   "num_colors",  "cidx"};
    for (const auto &item : kwargs) {
        const std::string key = py::cast<std::string>(item.first);
        if (key == "label") o.label = py::cast<std::string>(item.second);
        else if (key == "color") o.color = color_of(item.second);
        else if (key == "facecolor") o.facecolor = color_of(item.second);
        else if (key == "face_alpha") o.face_alpha = py::cast<double>(item.second);
        else if (key == "line_width") o.line_width = py::cast<double>(item.second);
        else if (key == "unify") o.unify = py::cast<bool>(item.second);
        else if (key == "filled") o.filled = py::cast<bool>(item.second);
        else if (key == "time_points") o.time_points = py::cast<bool>(item.second);
        else if (key == "step") o.step = py::cast<int>(item.second);
        else if (key == "num_colors") o.num_colors = py::cast<int>(item.second);
        else if (key == "cidx") o.cidx = py::cast<int>(item.second);
        else
            CORA_FAIL(Error::redundant_name_value_pair,
                      "cora.plot: unknown option '" + key + "'" + did_you_mean(key, known));
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
    }
    py::list polygons;
    for (const Polygon &p : l.polygons) polygons.append(polygon_array(p));
    d["polygons"] = polygons;
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
        for (const torch::Tensor &t : py::cast<torch::Tensor>(o).unbind(0)) x.push_back(from_torch(t));
        return x;
    }
#endif
    for (const py::handle &step : o) x.push_back(from_eigen(py::cast<Eigen::MatrixXd>(step)));
    return x;
}

void bind_plot(py::module_ &m) {
    py::class_<Figure>(m, "Figure", "The layers that plot puts into it, with every choice made.")
        .def(py::init<>())
        .def_readwrite("title", &Figure::title)
        .def_readwrite("xlabel", &Figure::xlabel)
        .def_readwrite("ylabel", &Figure::ylabel)
        .def_readwrite("equal_axes", &Figure::equal_axes)
        .def(
            "plot",
            [](Figure &fig, const ContSet &S, const std::vector<int64_t> &dims,
               const py::kwargs &kwargs) { plot(fig, S, dims, options_of(kwargs)); },
            py::arg("obj"), py::arg("dims"), "Draws a set, projected onto the two dimensions dims")
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
                if (ndim == 3) plot(fig, trajectories(data), dims, options_of(kwargs));
                else plot(fig, tensor_of(data, "the points"), dims, options_of(kwargs));
            },
            py::arg("obj"), py::arg("dims"), "Draws points (n, N) or simulations (steps, n, N)")
        .def("take_layers",
             [](Figure &fig) {
                 py::list layers;
                 for (const Figure::Layer &l : fig.take_layers()) layers.append(layer_dict(l));
                 return layers;
             },
             "The layers drawn since the last call, as dicts, and forgets them")
        .def_static(
            "clip_region",
            [](const py::array_t<double, py::array::c_style | py::array::forcecast> &polygon,
               const std::array<double, 2> &a, double b, double sign) {
                Polygon corners;
                auto view = polygon.unchecked<2>();
                for (py::ssize_t i = 0; i < view.shape(0); ++i) corners.push_back({view(i, 0), view(i, 1)});
                return polygon_array(Figure::clip_region(corners, a, b, sign));
            },
            py::arg("polygon"), py::arg("a"), py::arg("b"), py::arg("sign"),
            "The part of a polygon where sign * (a.x - b) >= 0");

    m.def("use_cora_colors", &use_cora_colors, py::arg("scheme"),
          "'CORA:default' (the next color of CORA's order for every set) or 'CORA:contDynamics'");
    m.def(
        "cora_color",
        [](const std::string &identifier, int num_colors, int cidx, double alpha) {
            return rgb(cora_color(identifier, num_colors, cidx, alpha));
        },
        py::arg("identifier"), py::arg("num_colors") = 1, py::arg("cidx") = 1, py::arg("alpha") = 0.2,
        "The RGB triple of a CORA color such as 'CORA:reachSet'");
    m.def(
        "cora_color",
        [](int number) { return rgb(cora_color("CORA:color" + std::to_string(number))); },
        py::arg("number"), "The RGB triple of the number-th color of CORA's palette");
}

void bind_specification(py::module_ &m) {
    py::class_<Specification>(m, "Specification",
                              "Halfspaces {x | a.x <= b} a set must stay in (safe_set) or must not "
                              "touch (unsafe_set).")
        .def_static("safe_set",
                    [](const py::object &a, double b) {
                        return Specification::safe_set(tensor_of(a, "the normal"), b);
                    },
                    py::arg("a"), py::arg("b"))
        .def_static("unsafe_set",
                    [](const py::object &a, const py::object &b) {
                        if (is_number(b))
                            return Specification::unsafe_set(tensor_of(a, "the normal"), py::cast<double>(b));
                        // the rows of A with b: a polytope the sets must not touch
                        const Eigen::MatrixXd A = py::cast<Eigen::MatrixXd>(a);
                        const Eigen::VectorXd bs = py::cast<Eigen::VectorXd>(b);
                        std::vector<Halfspace> halfspaces;
                        for (Eigen::Index i = 0; i < A.rows(); ++i)
                            halfspaces.push_back({column(Eigen::VectorXd(A.row(i))), bs(i)});
                        return Specification::unsafe_set(std::move(halfspaces));
                    },
                    py::arg("a"), py::arg("b"),
                    "One halfspace a.x <= b, or a polytope A x <= b (A a matrix, b a vector)")
        .def_property_readonly("type",
                               [](const Specification &spec) {
                                   return spec.type() == SpecType::safe_set ? "safe_set" : "unsafe_set";
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
        .def("holds", [](const Specification &spec, const ContSet &S) { return spec.holds(S); },
             py::arg("S"), "One flag per batch element of the set")
        .def("check", [](const Specification &spec, const ContSet &S) { return spec.check(S); },
             py::arg("S"), "Whether every batch element of the set satisfies the specification")
        .def("check",
             [](const Specification &spec, const std::vector<Zonotope> &sets) { return spec.check(sets); },
             py::arg("sets"), "Whether every set of a list (such as R.time_interval) does")
        .def("first_violation",
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
    m.def("dtype", [](const std::string &spec) { return new DType{spec, std::nullopt}; },
          py::arg("spec"),
          "with cora.lean.dtype('ieee:binary32'): new lean objects in that dtype in the block; "
          "'binary64', 'ieee:<format>', 'dyadic:<p>', 'fixedpoint:<f>', 'float', 'nearest'");
    m.def("set_default_dtype", &lean::set_default_dtype, py::arg("spec"),
          "The dtype of threads without a cora.lean.dtype block");
    m.def("current_dtype", &lean::dtype, "This thread's dtype");

    py::class_<lean::Tensor>(m, "Tensor", "A matrix of exact values in a dtype")
        .def_static("from_array", [](const Eigen::MatrixXd &x) { return lean::Tensor::from(from_eigen(x)); },
                    py::arg("x"), "Exact, or an error if a value is not representable in the dtype")
        .def("gather", [](const lean::Tensor &T) { return matrix(T.gather()); },
             "The values as an array; an error if one is not a double")
        .def(
            "round_to",
            [](const lean::Tensor &T, const std::string &dtype, const std::string &mode) {
                auto [value, error] = T.round_to(dtype, mode);
                return std::make_pair(value, error);
            },
            py::arg("dtype"), py::arg("mode") = "nearest",
            "(value, error): the rounded values and an interval enclosing x - value")
        .def_property_readonly("dtype", &lean::Tensor::dtype);

    py::class_<lean::Interval>(m, "Interval", "A box of lean values")
        .def(py::init<lean::Tensor, lean::Tensor>(), py::arg("inf"), py::arg("sup"))
        .def_static(
            "from_arrays",
            [](const Eigen::VectorXd &inf, const Eigen::VectorXd &sup) {
                return lean::Interval::from(column(inf), column(sup));
            },
            py::arg("inf"), py::arg("sup"))
        .def("gather",
             [](const lean::Interval &I) {
                 auto [inf, sup] = I.gather();
                 return Interval(inf, sup);
             },
             "A cora.Interval; an error if a bound is not a double");

    py::class_<lean::Zonotope>(m, "Zonotope", "A nominal zonotope plus an error box")
        .def(py::init([](const Eigen::VectorXd &c, const Eigen::MatrixXd &G) {
                 return lean::Zonotope(column(c), from_eigen(G));
             }),
             py::arg("center"), py::arg("generators"), "An exact zonotope in the current dtype")
        .def("nominal", &lean::Zonotope::nominal, "c and G with an empty error box")
        .def("error", &lean::Zonotope::error, "The error box")
        .def("enclosure", &lean::Zonotope::enclosure, "nominal and error as one zonotope")
        .def("round_to", &lean::Zonotope::round_to, py::arg("dtype"), py::arg("mode") = "nearest")
        .def("linear_map",
             [](const lean::Zonotope &Z, const py::object &M) {
                 if (py::isinstance<lean::Interval>(M)) return Z.linear_map(py::cast<lean::Interval>(M));
                 return Z.linear_map(lean::Tensor::from(from_eigen(py::cast<Eigen::MatrixXd>(M))));
             },
             py::arg("M"))
        .def("plus", py::overload_cast<const lean::Zonotope &>(&lean::Zonotope::plus, py::const_),
             py::arg("Z2"))
        .def("plus", py::overload_cast<const lean::Interval &>(&lean::Zonotope::plus, py::const_),
             py::arg("I"))
        .def("linear_combination", &lean::Zonotope::linear_combination, py::arg("Z2"))
        .def("reduce", &lean::Zonotope::reduce, py::arg("order"), py::arg("method") = "girard")
        .def("interval", &lean::Zonotope::interval)
        .def("gather", [](const lean::Zonotope &Z) { return Z.gather(); },
             "(cora.Zonotope, cora.Interval): the nominal part and the error, exactly");

    py::class_<lean::ReachSet>(m, "ReachSet", "The reachable sets: lists of lean zonotopes")
        .def_readonly("time_interval", &lean::ReachSet::time_interval)
        .def_readonly("time_point", &lean::ReachSet::time_point);

    py::class_<lean::LinearSys>(m, "LinearSys", "The linear system x' = A x, reached by CORALean")
        .def(py::init([](const Eigen::MatrixXd &A) { return lean::LinearSys(from_eigen(A)); }),
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
    set_default_backend("eigen");  // the Python package has no torch tensors, whatever the library has
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
