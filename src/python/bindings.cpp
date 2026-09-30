// bindings - the compiled half of the `cora` Python package (`_cora`)
//
// The classes are the C++ classes, with the same names and methods, so code reads the same in
// both languages and a batch lives in the object, not in the call:
//
//     X0  = cora.Zonotope(c, G)                 # c (..., n), G (..., n, m): one set or a batch
//     sys = cora.LinearSys(A)                   # A (..., n, n): one system or a batch
//     R   = sys.reach(X0, timeStep=0.1, tFinal=2.0)
//     R.timeInt[k].c
//
// Torch tensors (when built against torch, CORACPP_PYTHON_TORCH) run on the libtorch backend
// (batched, on any device, differentiable), numpy arrays on Eigen; an object keeps its backend and hands its numbers back the same way.
// A vector is 1-D here, (..., n); in C++ it is a column (..., n, 1).
//
// Syntax:   cora.Zonotope, Interval, LinearSys, NonlinearSys, NeuralNetwork, Expr, Reach,
//           Specification, Rng, setBackend
// See also: the C++ headers, which document every method

#include "contDynamics/linearSys/linearSys.h"
#include "contDynamics/nonlinearSys/nonlinearSys.h"
#include "global/rng.h"
#include "lean/linearSys.h"
#include "lean/oracle.h"
#include "global/plot/plot.h"
#include "nn/neuralNetwork/neuralNetwork.h"
#include "specification/specification.h"
#include "tensor/eigen.h"

#include <optional>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#ifdef CORACPP_PYTHON_TORCH
#include "tensor/torch.h"
#include <torch/extension.h>
#endif

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace py = pybind11;
using namespace cora;

namespace {

// ---------------- vectors and matrices between Python and CoraTensor -------------------

/// A Python vector (..., n) as a column (..., n, 1).
#ifdef CORACPP_PYTHON_TORCH
Tensor column(const torch::Tensor &v) { return fromTorch(v.unsqueeze(-1)); }
#endif
Tensor column(const Eigen::VectorXd &v) { return fromEigen(v); }

/// A column (..., n, 1) as a Python vector (..., n), in the array type of its backend.
py::object vector(const Tensor &t) {
#ifdef CORACPP_PYTHON_TORCH
    if (isTorch(t)) return py::cast(toTorch(t).squeeze(-1));
#endif
    if (isEigen(t)) return py::cast(Eigen::VectorXd(toEigen(t).col(0)));
    throw std::invalid_argument("cora: a tensor of an unknown backend");
}

/// A matrix, or the bounds of an interval, in the array type of its backend: a column bound is
/// a vector (..., n), a matrix bound stays (..., n, n).
py::object matrix(const Tensor &t) {
#ifdef CORACPP_PYTHON_TORCH
    if (isTorch(t)) return py::cast(toTorch(t));
#endif
    if (isEigen(t)) return py::cast(toEigen(t));
    throw std::invalid_argument("cora: a tensor of an unknown backend");
}

py::object bounds(const Tensor &t) {
    return t.shape().back() == 1 ? vector(t) : matrix(t);
}

/// A (..., 1, 1) result as a scalar per batch element: a torch tensor (...) or a float.
py::object scalar(const Tensor &t) {
#ifdef CORACPP_PYTHON_TORCH
    if (isTorch(t)) return py::cast(toTorch(t).squeeze(-1).squeeze(-1));
#endif
    if (isEigen(t)) return py::cast(toEigen(t)(0, 0));
    throw std::invalid_argument("cora: a tensor of an unknown backend");
}

/// A torch tensor or numpy array from Python as a Tensor on its backend (a 1-D one is a column);
/// nothing for anything else.
std::optional<Tensor> asTensor(const py::object &o) {
#ifdef CORACPP_PYTHON_TORCH
    py::detail::make_caster<torch::Tensor> torchCaster;
    if (torchCaster.load(o, /*convert=*/false)) {
        const torch::Tensor t = o.cast<torch::Tensor>();
        return t.dim() == 1 ? column(t) : fromTorch(t);
    }
#endif
    if (py::isinstance<py::array>(o)) {
        const py::array a = py::array::ensure(o);
        if (a.ndim() == 1) return column(o.cast<Eigen::VectorXd>());
        return fromEigen(o.cast<Eigen::MatrixXd>());
    }
    return std::nullopt;
}

/// The time points of a simulation as one array, the time first: (steps + 1, ..., n, N).
py::object stack(const std::vector<Tensor> &points) {
    if (points.empty()) throw std::invalid_argument("cora: no time points");
#ifdef CORACPP_PYTHON_TORCH
    if (isTorch(points[0])) {
        std::vector<torch::Tensor> all;
        for (const Tensor &x : points) all.push_back(toTorch(x));
        return py::cast(torch::stack(torch::broadcast_tensors(all), 0));
    }
#endif
    py::list all;
    for (const Tensor &x : points) all.append(py::cast(toEigen(x)));
    return py::module_::import("numpy").attr("stack")(all);
}

/// CORA's linAlg: "standard" or "wrapping-free".
Algorithm parseLinAlg(const std::string &linAlg) {
    if (linAlg == "standard") return Algorithm::Standard;
    if (linAlg == "wrapping-free") return Algorithm::WrappingFree;
    throw py::value_error("unknown linAlg '" + linAlg + "'; use 'standard' or 'wrapping-free'");
}

/// A polygon as an (k, 2) array.
py::array_t<double> polygonArray(const Polygon &polygon) {
    py::array_t<double> out({static_cast<py::ssize_t>(polygon.size()), py::ssize_t(2)});
    auto view = out.mutable_unchecked<2>();
    for (std::size_t i = 0; i < polygon.size(); ++i) {
        view(i, 0) = polygon[i][0];
        view(i, 1) = polygon[i][1];
    }
    return out;
}

// --------------------------------- classes -----------------------------------------

/// The operations every set has; bound once on the base class, so any set works wherever a set
/// is expected.
void bindContSet(py::module_ &m) {
    py::class_<ContSet>(m, "ContSet", "The abstract set: what every set answers.")
        .def("dim", &ContSet::dim)
        .def("center", [](const ContSet &S) { return vector(S.center()); })
        .def("interval", &ContSet::interval)
        .def("vertices",
             [](const ContSet &S) {
                 py::list polygons;
                 for (const Polygon &p : S.vertices()) polygons.append(polygonArray(p));
                 return polygons;
             },
             "The vertices of a two-dimensional set, one (k, 2) array per batch member")
        .def("__repr__", &ContSet::display)
        .def("__str__", &ContSet::display)
#ifdef CORACPP_PYTHON_TORCH
        .def("supportFunc",
             [](const ContSet &S, const torch::Tensor &d) { return scalar(S.supportFunc(column(d))); },
             py::arg("d"), "max of d'x over the set; d (..., n)")
#endif
        .def("supportFunc",
             [](const ContSet &S, const Eigen::VectorXd &d) { return scalar(S.supportFunc(column(d))); },
             py::arg("d"))
        .def("randPoint",
             [](const ContSet &S, int64_t N, cora::Rng &rng) { return matrix(S.randPoint(N, rng)); },
             py::arg("N"), py::arg("rng"), "N random points as columns (..., n, N)");
}

void bindInterval(py::module_ &m) {
    py::class_<Interval, ContSet>(m, "Interval", "A box, or a matrix of intervals: inf <= x <= sup.")
#ifdef CORACPP_PYTHON_TORCH
        .def(py::init([](const torch::Tensor &inf, const torch::Tensor &sup) {
                 return Interval(column(inf), column(sup));
             }),
             py::arg("inf"), py::arg("sup"))
#endif
        .def(py::init([](const Eigen::VectorXd &inf, const Eigen::VectorXd &sup) {
                 return Interval(column(inf), column(sup));
             }),
             py::arg("inf"), py::arg("sup"))
#ifdef CORACPP_PYTHON_TORCH
        .def_static("matrix",
                    [](const torch::Tensor &inf, const torch::Tensor &sup) {
                        return Interval(fromTorch(inf), fromTorch(sup));
                    },
                    py::arg("inf"), py::arg("sup"),
                    "An interval matrix: bounds (..., n, n) are taken as they are, not as vectors")
#endif
        .def_static("matrix",
                    [](const Eigen::MatrixXd &inf, const Eigen::MatrixXd &sup) {
                        return Interval(fromEigen(inf), fromEigen(sup));
                    },
                    py::arg("inf"), py::arg("sup"))
        .def_static("stack", &Interval::stack, py::arg("intervals"),
                    "A batch of intervals as one; the batch is the leading dimension")
        .def_static(
            "generateRandom",
            [](int64_t n, cora::Rng &rng) { return Interval::generateRandom(n, rng); }, py::arg("n"),
            py::arg("rng"))
        .def_property_readonly("inf", [](const Interval &I) { return bounds(I.inf); })
        .def_property_readonly("sup", [](const Interval &I) { return bounds(I.sup); })
        .def("rad", [](const Interval &I) { return bounds(I.rad()); })
#ifdef CORACPP_PYTHON_TORCH
        .def("mtimes", [](const Interval &I, const torch::Tensor &M) { return I.mtimes(fromTorch(M)); },
             py::arg("M"))
#endif
        .def("mtimes", [](const Interval &I, const Eigen::MatrixXd &M) { return I.mtimes(fromEigen(M)); },
             py::arg("M"))
        .def("plus", &Interval::plus, py::arg("I2"))
#ifdef CORACPP_PYTHON_TORCH
        .def("contains", [](const Interval &I, const torch::Tensor &p) { return I.contains(column(p)); },
             py::arg("p"))
#endif
        .def("contains", [](const Interval &I, const Eigen::VectorXd &p) { return I.contains(column(p)); },
             py::arg("p"));
}

void bindZonotope(py::module_ &m) {
    py::class_<Zonotope, ContSet>(m, "Zonotope", "The zonotope {c + G b : |b|_inf <= 1}.")
#ifdef CORACPP_PYTHON_TORCH
        .def(py::init([](const torch::Tensor &c, const torch::Tensor &G) {
                 return Zonotope(column(c), fromTorch(G));
             }),
             py::arg("c"), py::arg("G"), "center c (..., n) and generators G (..., n, m)")
#endif
        .def(py::init([](const Eigen::VectorXd &c, const Eigen::MatrixXd &G) {
                 return Zonotope(column(c), fromEigen(G));
             }),
             py::arg("c"), py::arg("G"))
        .def_static("stack", &Zonotope::stack, py::arg("zonotopes"),
                    "A batch of zonotopes as one; fewer generators are padded with zeros")
        .def_static(
            "generateRandom",
            [](int64_t n, int64_t m, cora::Rng &rng) { return Zonotope::generateRandom(n, m, rng); },
            py::arg("n"), py::arg("m"), py::arg("rng"))
        .def_property_readonly("c", [](const Zonotope &Z) { return vector(Z.c); })
        .def_property_readonly("G", [](const Zonotope &Z) { return matrix(Z.G); })
        .def(
            "randPoint",
            [](const Zonotope &Z, int64_t N, cora::Rng &rng, const std::string &type) {
                return matrix(Z.randPoint(N, rng, type));
            },
            py::arg("N"), py::arg("rng"), py::arg("type") = "standard",
            "N random points as columns (..., n, N); type 'standard' or 'extreme'")
#ifdef CORACPP_PYTHON_TORCH
        .def("mtimes", [](const Zonotope &Z, const torch::Tensor &M) { return Z.mtimes(fromTorch(M)); },
             py::arg("M"), "M * Z for a matrix M (..., n, n)")
#endif
        .def("mtimes", [](const Zonotope &Z, const Eigen::MatrixXd &M) { return Z.mtimes(fromEigen(M)); },
             py::arg("M"))
        .def("mtimes", [](const Zonotope &Z, const Interval &M) { return Z.mtimes(M); }, py::arg("M"),
             "[M] * Z for an interval matrix")
        .def("plus", &Zonotope::plus, py::arg("Z2"))
        .def("linComb", &Zonotope::linComb, py::arg("Z2"))
        .def("reduce", &Zonotope::reduce, py::arg("order"),
             "At most order * n generators; the smallest are enclosed by a box");
}

void bindLinearSys(py::module_ &m) {
    py::class_<Reach>(m, "Reach", "The reachable sets: lists of zonotopes.")
        .def_readonly("timeInt", &Reach::timeInt, "timeInt[k] over [k*timeStep, (k+1)*timeStep]")
        .def_readonly("timePoint", &Reach::timePoint, "timePoint[k] at k*timeStep");

    py::class_<LinearSys>(m, "LinearSys", "The linear system x' = A x.")
#ifdef CORACPP_PYTHON_TORCH
        .def(py::init([](const torch::Tensor &A, bool customBackward) {
                 return LinearSys(fromTorch(A, customBackward));
             }),
             py::arg("A"), py::arg("customBackward") = false,
             "A (..., n, n); customBackward: hand-written backward pass for the matrix exponential")
#endif
        .def(py::init([](const Eigen::MatrixXd &A) { return LinearSys(fromEigen(A)); }), py::arg("A"))
        .def_property_readonly("A", [](const LinearSys &sys) { return matrix(sys.A()); })
        .def(
            "reach",
            [](const LinearSys &sys, const Zonotope &X0, double timeStep, double tFinal,
               int taylorTerms, const std::string &linAlg) {
                return sys.reach(X0, timeStep, tFinal, taylorTerms, parseLinAlg(linAlg));
            },
            py::arg("X0"), py::arg("timeStep"), py::arg("tFinal"), py::arg("taylorTerms") = 10,
            py::arg("linAlg") = "standard", "The reachable sets from the zonotope X0")
#ifdef CORACPP_PYTHON_TORCH
        .def(
            "simulate",
            [](const LinearSys &sys, const torch::Tensor &x0, double timeStep, double tFinal) {
                return stack(sys.simulate(fromTorch(x0), timeStep, tFinal));
            },
            py::arg("x0"), py::arg("timeStep"), py::arg("tFinal"),
            "Trajectories from the points x0 (..., n, N): (steps + 1, ..., n, N), time first")
#endif
        .def(
            "simulate",
            [](const LinearSys &sys, const Eigen::MatrixXd &x0, double timeStep, double tFinal) {
                return stack(sys.simulate(fromEigen(x0), timeStep, tFinal));
            },
            py::arg("x0"), py::arg("timeStep"), py::arg("tFinal"))
        .def(
            "simulateRandom",
            [](const LinearSys &sys, const ContSet &X0, int64_t N, double timeStep, double tFinal,
               cora::Rng &rng) { return stack(sys.simulateRandom(X0, N, timeStep, tFinal, rng)); },
            py::arg("X0"), py::arg("N"), py::arg("timeStep"), py::arg("tFinal"), py::arg("rng"),
            "simulate from N random points of the set X0")
        .def("correctionMatrixState", &LinearSys::correctionMatrixState, py::arg("timeStep"),
             py::arg("taylorTerms"), "The interval matrix F of the curvature enlargement");
}

/// The dynamics as symbolic expressions: `x` is a list of Expr, `f(x)` a list of Expr or numbers.
void bindNonlinearSys(py::module_ &m) {
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
        .def("dim", &NonlinearSys::dim)
        .def("reach", &NonlinearSys::reach, py::arg("X0"), py::arg("timeStep"), py::arg("tFinal"),
             py::arg("taylorTerms") = 4, py::arg("zonotopeOrder") = 50,
             "The reachable sets from the zonotope X0 (algorithm 'lin')")
#ifdef CORACPP_PYTHON_TORCH
        .def("simulate",
             [](const NonlinearSys &sys, const torch::Tensor &x0, double timeStep, double tFinal) {
                 return stack(sys.simulate(fromTorch(x0), timeStep, tFinal));
             },
             py::arg("x0"), py::arg("timeStep"), py::arg("tFinal"),
             "Trajectories from the points x0 (n, N): (steps + 1, n, N), time first")
#endif
        .def("simulate",
             [](const NonlinearSys &sys, const Eigen::MatrixXd &x0, double timeStep, double tFinal) {
                 return stack(sys.simulate(fromEigen(x0), timeStep, tFinal));
             },
             py::arg("x0"), py::arg("timeStep"), py::arg("tFinal"))
        .def("simulateRandom",
             [](const NonlinearSys &sys, const ContSet &X0, int64_t N, double timeStep, double tFinal,
                cora::Rng &rng) { return stack(sys.simulateRandom(X0, N, timeStep, tFinal, rng)); },
             py::arg("X0"), py::arg("N"), py::arg("timeStep"), py::arg("tFinal"), py::arg("rng"),
             "simulate from N random points of the set X0");
}

void bindNeuralNetwork(py::module_ &m) {
    py::class_<NeuralNetwork>(m, "NeuralNetwork",
                              "Affine layers with a ReLU after each but the last; torch tensors.")
#ifdef CORACPP_PYTHON_TORCH
        .def(py::init([](const std::vector<std::pair<torch::Tensor, torch::Tensor>> &layers) {
                 std::vector<NeuralNetwork::Layer> out;
                 for (const auto &[W, b] : layers)
                     out.push_back({fromTorch(W), b.dim() == 1 ? column(b) : fromTorch(b)});
                 return NeuralNetwork(out);
             }),
             py::arg("layers"), "the layers as (W (out, in), b (out,)) pairs, first to last")
#endif
        .def_property_readonly("inputDim", &NeuralNetwork::inputDim)
        .def_property_readonly("outputDim", &NeuralNetwork::outputDim)
#ifdef CORACPP_PYTHON_TORCH
        .def(
            "evaluate",
            [](const NeuralNetwork &nn, const torch::Tensor &x) {
                if (x.dim() == 1) return toTorch(nn.evaluate(column(x))).squeeze(-1);
                return toTorch(nn.evaluate(fromTorch(x)));
            },
            py::arg("x"), "The outputs at a point (in,) or at the columns of (in, N)")
#endif
        .def("evaluate", [](const NeuralNetwork &nn, const Zonotope &X) { return nn.evaluate(X); },
             py::arg("X"), "A zonotope that contains the outputs of every point of X");
}

/// The operators of the sets, as CORA writes them: `A * Z`, `s * Z`, `Z + Z2`, `Z + v`, `Z - v`, `-Z`.
/// A torch tensor or numpy array is a matrix, or a vector when it is 1-D; what an operator does
/// not understand it hands back to Python (NotImplemented), which raises the TypeError.
void bindOperators(py::module_ &m) {
    const py::object Z = m.attr("Zonotope"), I = m.attr("Interval");
    Z.attr("__array_ufunc__") = py::none();  // numpy defers to the operators below
    I.attr("__array_ufunc__") = py::none();
    const auto method = [](const py::object &cls, const char *name, auto f) {
        cls.attr(name) = py::cpp_function(f, py::is_method(cls));
    };
    const auto number = [](const py::object &o) {
        return py::isinstance<py::float_>(o) || py::isinstance<py::int_>(o);
    };
    const auto refused = [] { return py::reinterpret_borrow<py::object>(Py_NotImplemented); };

    // Zonotope
    const auto zonotopeSum = [=](const Zonotope &A, const py::object &o) -> py::object {
        if (py::isinstance<Zonotope>(o)) return py::cast(A + o.cast<Zonotope>());
        if (const std::optional<Tensor> v = asTensor(o)) return py::cast(A + *v);
        return refused();
    };
    method(Z, "__add__", zonotopeSum);
    method(Z, "__radd__", zonotopeSum);
    method(Z, "__sub__", [=](const Zonotope &A, const py::object &o) -> py::object {
        if (const std::optional<Tensor> v = asTensor(o)) return py::cast(A - *v);
        return refused();
    });
    method(Z, "__neg__", [](const Zonotope &A) { return -A; });
    method(Z, "__mul__", [=](const Zonotope &A, const py::object &o) -> py::object {
        if (number(o)) return py::cast(A * o.cast<double>());
        return refused();
    });
    const auto zonotopeProduct = [=](const Zonotope &A, const py::object &o) -> py::object {
        if (number(o)) return py::cast(o.cast<double>() * A);
        if (const std::optional<Tensor> M = asTensor(o)) return py::cast(*M * A);
        return refused();
    };
    method(Z, "__rmul__", zonotopeProduct);
    method(Z, "__rmatmul__", zonotopeProduct);

    // Interval
    const auto intervalSum = [=](const Interval &A, const py::object &o) -> py::object {
        if (py::isinstance<Interval>(o)) return py::cast(A + o.cast<Interval>());
        if (const std::optional<Tensor> v = asTensor(o)) return py::cast(A + *v);
        return refused();
    };
    method(I, "__add__", intervalSum);
    method(I, "__radd__", intervalSum);
    method(I, "__sub__", [=](const Interval &A, const py::object &o) -> py::object {
        if (const std::optional<Tensor> v = asTensor(o)) return py::cast(A - *v);
        return refused();
    });
    method(I, "__neg__", [](const Interval &A) { return -A; });
    method(I, "__mul__", [=](const Interval &A, const py::object &o) -> py::object {
        if (number(o)) return py::cast(A * o.cast<double>());
        if (py::isinstance<Zonotope>(o)) return py::cast(A * o.cast<Zonotope>());
        return refused();
    });
    const auto intervalProduct = [=](const Interval &A, const py::object &o) -> py::object {
        if (number(o)) return py::cast(o.cast<double>() * A);
        if (const std::optional<Tensor> M = asTensor(o)) return py::cast(*M * A);
        return refused();
    };
    method(I, "__rmul__", intervalProduct);
    method(I, "__rmatmul__", intervalProduct);
}

// --------------------------------- plotting -----------------------------------------
// The plotting logic is the C++ one (plot/plot.h): a Figure collects layers with every choice
// made, and the Python plot draws them with matplotlib.

/// A color from Python: a CORA identifier ("CORA:red") or an RGB triple.
Color colorOf(const py::handle &value) {
    if (py::isinstance<py::str>(value)) return Color(value.cast<std::string>());
    const std::vector<double> rgb = value.cast<std::vector<double>>();
    if (rgb.size() != 3)
        throw std::invalid_argument("cora.plot: a color is a CORA identifier such as "
                                    "'CORA:red' or an (r, g, b) triple");
    return Color(rgb[0], rgb[1], rgb[2]);
}

/// The plot options given as keyword arguments, named as in PlotOptions.
PlotOptions optionsOf(const py::kwargs &kwargs) {
    PlotOptions o;
    for (const auto &item : kwargs) {
        const std::string key = item.first.cast<std::string>();
        if (key == "label") o.label = item.second.cast<std::string>();
        else if (key == "color") o.color = colorOf(item.second);
        else if (key == "facecolor") o.facecolor = colorOf(item.second);
        else if (key == "faceAlpha") o.faceAlpha = item.second.cast<double>();
        else if (key == "lineWidth") o.lineWidth = item.second.cast<double>();
        else if (key == "unify") o.unify = item.second.cast<bool>();
        else if (key == "filled") o.filled = item.second.cast<bool>();
        else if (key == "timePoints") o.timePoints = item.second.cast<bool>();
        else if (key == "step") o.step = item.second.cast<int>();
        else if (key == "numColors") o.numColors = item.second.cast<int>();
        else if (key == "cidx") o.cidx = item.second.cast<int>();
        else
            throw std::invalid_argument("cora.plot: unknown option '" + key + "'; the options are "
                                        "label, color, facecolor, faceAlpha, lineWidth, unify, filled, "
                                        "timePoints, step, numColors and cidx");
    }
    return o;
}

py::tuple rgb(const Color &c) { return py::make_tuple(c.r, c.g, c.b); }

/// A layer of a figure as a dict, for the code that draws it.
py::dict layerDict(const Figure::Layer &l) {
    py::dict d;
    switch (l.kind) {
    case Figure::Kind::Polygons: d["kind"] = "polygons"; break;
    case Figure::Kind::Polyline: d["kind"] = "polyline"; break;
    case Figure::Kind::Points: d["kind"] = "points"; break;
    case Figure::Kind::Region: d["kind"] = "region"; break;
    }
    py::list polygons;
    for (const Polygon &p : l.polygons) polygons.append(polygonArray(p));
    d["polygons"] = polygons;
    d["edge"] = rgb(l.edge);
    d["face"] = l.filled || l.kind == Figure::Kind::Region ? py::object(rgb(l.face)) : py::none();
    d["unify"] = l.unify;
    d["faceAlpha"] = l.faceAlpha;
    d["dashed"] = l.dashed;
    d["lineWidth"] = l.lineWidth;
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
    py::detail::make_caster<torch::Tensor> torchCaster;
    if (torchCaster.load(o, /*convert=*/false)) {
        for (const torch::Tensor &t : o.cast<torch::Tensor>().unbind(0)) x.push_back(fromTorch(t));
        return x;
    }
    for (const py::handle &step : o) x.push_back(fromEigen(step.cast<Eigen::MatrixXd>()));
    return x;
}

void bindPlot(py::module_ &m) {
    py::class_<Figure>(m, "Figure", "The layers that plot puts into it, with every choice made.")
        .def(py::init<>())
        .def_readwrite("title", &Figure::title)
        .def_readwrite("xlabel", &Figure::xlabel)
        .def_readwrite("ylabel", &Figure::ylabel)
        .def_readwrite("equalAxes", &Figure::equalAxes)
        .def(
            "plot",
            [](Figure &fig, const ContSet &S, const std::vector<int64_t> &dims,
               const py::kwargs &kwargs) { plot(fig, S, dims, optionsOf(kwargs)); },
            py::arg("obj"), py::arg("dims"), "Draws a set, projected onto the two dimensions dims")
        .def(
            "plot",
            [](Figure &fig, const Reach &R, const std::vector<int64_t> &dims,
               const py::kwargs &kwargs) { plot(fig, R, dims, optionsOf(kwargs)); },
            py::arg("obj"), py::arg("dims"))
        .def(
            "plot",
            [](Figure &fig, const Specification &spec, const std::vector<int64_t> &dims,
               const py::kwargs &kwargs) { plot(fig, spec, dims, optionsOf(kwargs)); },
            py::arg("obj"), py::arg("dims"))
        .def(
            "plot",
            [](Figure &fig, const py::object &data, const std::vector<int64_t> &dims,
               const py::kwargs &kwargs) {
                // Points (n, N) or the time points of a simulation (steps, n, N).
                const py::ssize_t ndim = py::hasattr(data, "ndim") ? data.attr("ndim").cast<py::ssize_t>()
                                                                   : data.attr("dim")().cast<py::ssize_t>();
                if (ndim == 3) plot(fig, trajectories(data), dims, optionsOf(kwargs));
                else if (const std::optional<Tensor> points = asTensor(data))
                    plot(fig, *points, dims, optionsOf(kwargs));
                else
                    throw std::invalid_argument("cora.plot: the data must be a torch tensor or a "
                                                "numpy array");
            },
            py::arg("obj"), py::arg("dims"), "Draws points (n, N) or simulations (steps, n, N)")
        .def("takeLayers",
             [](Figure &fig) {
                 py::list layers;
                 for (const Figure::Layer &l : fig.takeLayers()) layers.append(layerDict(l));
                 return layers;
             },
             "The layers drawn since the last call, as dicts, and forgets them")
        .def_static(
            "clipRegion",
            [](const py::array_t<double, py::array::c_style | py::array::forcecast> &polygon,
               const std::array<double, 2> &a, double b, double sign) {
                Polygon corners;
                auto view = polygon.unchecked<2>();
                for (py::ssize_t i = 0; i < view.shape(0); ++i) corners.push_back({view(i, 0), view(i, 1)});
                return polygonArray(Figure::clipRegion(corners, a, b, sign));
            },
            py::arg("polygon"), py::arg("a"), py::arg("b"), py::arg("sign"),
            "The part of a polygon where sign * (a.x - b) >= 0");

    m.def("useCORAcolors", &useCORAcolors, py::arg("scheme"),
          "'CORA:default' (the next color of CORA's order for every set) or 'CORA:contDynamics'");
    m.def(
        "CORAcolor",
        [](const std::string &identifier, int numColors, int cidx, double alpha) {
            return rgb(CORAcolor(identifier, numColors, cidx, alpha));
        },
        py::arg("identifier"), py::arg("numColors") = 1, py::arg("cidx") = 1, py::arg("alpha") = 0.2,
        "The RGB triple of a CORA color such as 'CORA:reachSet'");
    m.def(
        "CORAcolor",
        [](int number) { return rgb(CORAcolor("CORA:color" + std::to_string(number))); },
        py::arg("number"), "The RGB triple of the number-th color of CORA's palette");
}

void bindSpecification(py::module_ &m) {
    py::class_<Specification>(m, "Specification",
                              "Halfspaces {x | a.x <= b} a set must stay in (safeSet) or must not touch "
                              "(unsafeSet).")
#ifdef CORACPP_PYTHON_TORCH
        .def_static("safeSet",
                    [](const torch::Tensor &a, double b) { return Specification::safeSet(column(a), b); },
                    py::arg("a"), py::arg("b"))
#endif
        .def_static("safeSet",
                    [](const Eigen::VectorXd &a, double b) { return Specification::safeSet(column(a), b); },
                    py::arg("a"), py::arg("b"))
#ifdef CORACPP_PYTHON_TORCH
        .def_static("unsafeSet",
                    [](const torch::Tensor &a, double b) { return Specification::unsafeSet(column(a), b); },
                    py::arg("a"), py::arg("b"))
#endif
        .def_static("unsafeSet",
                    [](const Eigen::VectorXd &a, double b) { return Specification::unsafeSet(column(a), b); },
                    py::arg("a"), py::arg("b"))
        .def_property_readonly("type",
                               [](const Specification &spec) {
                                   if (spec.type() == SpecType::SafeSet) return "safeSet";
                                   if (spec.type() == SpecType::UnsafeSet) return "unsafeSet";
                                   throw std::invalid_argument("cora: unknown specification type");
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
        .def("holds", [](const Specification &spec, const ContSet &S) { return spec.holds(S); }, py::arg("S"),
             "One flag per batch element of the set")
        .def("check", [](const Specification &spec, const ContSet &S) { return spec.check(S); }, py::arg("S"),
             "Whether every batch element of the set satisfies the specification")
        .def("check", [](const Specification &spec, const std::vector<Zonotope> &sets) { return spec.check(sets); },
             py::arg("sets"), "Whether every set of a list (such as R.timeInt) does")
        .def("firstViolation",
             [](const Specification &spec, const std::vector<Zonotope> &sets) { return spec.firstViolation(sets); },
             py::arg("sets"), "The index of the first violating set, or -1");
}

/// The oracle of CORALean as `cora.lean`: the same names as cora::lean; vectors are 1-D arrays.
void bindLean(py::module_ &top) {
    py::module_ m = top.def_submodule("lean", "Sound float sets computed by CORALean (needs CORACPP_ORACLE)");
    m.def("setDType", &lean::setDType, py::arg("spec"),
          "The dtype of new lean objects: 'binary64', 'ieee:<format>', 'dyadic:<p>', 'fixedpoint:<f>'");
    m.def("dtype", &lean::dtype, "The current dtype");

    py::class_<lean::Tensor>(m, "Tensor", "A matrix of exact values in a dtype")
        .def_static("fromArray", [](const Eigen::MatrixXd &x) { return lean::Tensor::from(fromEigen(x)); },
                    py::arg("x"), "Exact, or an error if a value is not representable in the dtype")
        .def("gather", [](const lean::Tensor &T) { return matrix(T.gather()); },
             "The values as an array; an error if one is not a double")
        .def(
            "roundTo",
            [](const lean::Tensor &T, const std::string &dtype, const std::string &mode) {
                auto [value, error] = T.roundTo(dtype, mode);
                return std::make_pair(value, error);
            },
            py::arg("dtype"), py::arg("mode") = "nearest",
            "(value, error): the rounded values and an interval enclosing x - value")
        .def_property_readonly("dtype", &lean::Tensor::dtype);

    py::class_<lean::Interval>(m, "Interval", "A box of lean values")
        .def(py::init<lean::Tensor, lean::Tensor>(), py::arg("inf"), py::arg("sup"))
        .def_static(
            "fromArrays",
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
                 return lean::Zonotope(column(c), fromEigen(G));
             }),
             py::arg("c"), py::arg("G"), "An exact zonotope in the current dtype")
        .def("nominal", &lean::Zonotope::nominal, "c and G with an empty error box")
        .def("error", &lean::Zonotope::error, "The error box")
        .def("enclosure", &lean::Zonotope::enclosure, "nominal and error as one zonotope")
        .def("roundTo", &lean::Zonotope::roundTo, py::arg("dtype"), py::arg("mode") = "nearest")
        .def("mtimes",
             [](const lean::Zonotope &Z, const Eigen::MatrixXd &M) {
                 return Z.mtimes(lean::Tensor::from(fromEigen(M)));
             },
             py::arg("M"))
        .def("mtimes", py::overload_cast<const lean::Interval &>(&lean::Zonotope::mtimes, py::const_),
             py::arg("M"))
        .def("plus", py::overload_cast<const lean::Zonotope &>(&lean::Zonotope::plus, py::const_),
             py::arg("Z2"))
        .def("plus", py::overload_cast<const lean::Interval &>(&lean::Zonotope::plus, py::const_),
             py::arg("I"))
        .def("linComb", &lean::Zonotope::linComb, py::arg("Z2"))
        .def("reduce", &lean::Zonotope::reduce, py::arg("order"), py::arg("method") = "girard")
        .def("interval", &lean::Zonotope::interval)
        .def("gather", [](const lean::Zonotope &Z) { return Z.gather(); },
             "(cora.Zonotope, cora.Interval): the nominal part and the error, exactly");

    py::class_<lean::ReachSet>(m, "ReachSet", "The reachable sets: lists of lean zonotopes")
        .def_readonly("timeInt", &lean::ReachSet::timeInt)
        .def_readonly("timePoint", &lean::ReachSet::timePoint);

    py::class_<lean::LinearSys>(m, "LinearSys", "The linear system x' = A x, reached by CORALean")
        .def(py::init([](const Eigen::MatrixXd &A) { return lean::LinearSys(fromEigen(A)); }),
             py::arg("A"))
        .def("reach", &lean::LinearSys::reach, py::arg("X0"), py::arg("timeStep"), py::arg("tFinal"),
             py::arg("taylorTerms"), py::arg("zonotopeOrder") = 10);
}

} // namespace

PYBIND11_MODULE(_cora, m) {
    m.doc() = "CORA.cpp: set-based computing on Eigen (numpy) or libtorch (torch)";

    py::class_<cora::Rng>(m, "Rng", "Random numbers; a seed makes a draw repeatable.")
        .def(py::init<uint64_t>(), py::arg("seed") = 0);
    m.def("setBackend", &setBackend, py::arg("spec"),
          "The backend of new tensors: 'eigen', 'torch', 'torch:cuda', optionally ',customBackward'");
    m.def("backend", [] { return backend().name(); }, "The name of the current backend");

    bindContSet(m);
    bindInterval(m);
    bindZonotope(m);
    bindLinearSys(m);
    bindNonlinearSys(m);
    bindNeuralNetwork(m);
    bindOperators(m);
    bindPlot(m);
    bindSpecification(m);
    bindLean(m);
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
