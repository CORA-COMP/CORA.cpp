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
// Torch tensors run on the libtorch backend (batched, on any device, differentiable), numpy
// arrays on Eigen; an object keeps its backend and hands its numbers back the same way.
// A vector is 1-D here, (..., n); in C++ it is a column (..., n, 1).
//
// Syntax:   cora.Zonotope, Interval, LinearSys, Reach, Specification, Rng, setBackend
// See also: the C++ headers, which document every method

#include "contDynamics/linearSys/linearSys.h"
#include "global/rng.h"
#include "specification/specification.h"
#include "tensor/eigen.h"
#include "tensor/torch.h"

#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace py = pybind11;
using namespace cora::ct;

namespace {

// ---------------- vectors and matrices between Python and CoraTensor -------------------

/// A Python vector (..., n) as a column (..., n, 1).
Tensor column(const torch::Tensor &v) { return fromTorch(v.unsqueeze(-1)); }
Tensor column(const Eigen::VectorXd &v) { return fromEigen(v); }

/// A column (..., n, 1) as a Python vector (..., n), in the array type of its backend.
py::object vector(const Tensor &t) {
    if (isTorch(t)) return py::cast(toTorch(t).squeeze(-1));
    if (isEigen(t)) return py::cast(Eigen::VectorXd(toEigen(t).col(0)));
    throw std::invalid_argument("cora: a tensor of an unknown backend");
}

/// A matrix, or the bounds of an interval, in the array type of its backend: a column bound is
/// a vector (..., n), a matrix bound stays (..., n, n).
py::object matrix(const Tensor &t) {
    if (isTorch(t)) return py::cast(toTorch(t));
    if (isEigen(t)) return py::cast(toEigen(t));
    throw std::invalid_argument("cora: a tensor of an unknown backend");
}

py::object bounds(const Tensor &t) {
    return t.shape().back() == 1 ? vector(t) : matrix(t);
}

/// A (..., 1, 1) result as a scalar per batch element: a torch tensor (...) or a float.
py::object scalar(const Tensor &t) {
    if (isTorch(t)) return py::cast(toTorch(t).squeeze(-1).squeeze(-1));
    if (isEigen(t)) return py::cast(toEigen(t)(0, 0));
    throw std::invalid_argument("cora: a tensor of an unknown backend");
}

/// The time points of a simulation as one array, the time first: (steps + 1, ..., n, N).
py::object stack(const std::vector<Tensor> &points) {
    if (points.empty()) throw std::invalid_argument("cora: no time points");
    if (isTorch(points[0])) {
        std::vector<torch::Tensor> all;
        for (const Tensor &x : points) all.push_back(toTorch(x));
        return py::cast(torch::stack(torch::broadcast_tensors(all), 0));
    }
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

// --------------------------------- classes -----------------------------------------

/// The operations every set has; bound once on the base class, so any set works wherever a set
/// is expected.
void bindContSet(py::module_ &m) {
    py::class_<ContSet>(m, "ContSet", "The abstract set: what every set answers.")
        .def("dim", &ContSet::dim)
        .def("center", [](const ContSet &S) { return vector(S.center()); })
        .def("interval", &ContSet::interval)
        .def("supportFunc",
             [](const ContSet &S, const torch::Tensor &d) { return scalar(S.supportFunc(column(d))); },
             py::arg("d"), "max of d'x over the set; d (..., n)")
        .def("supportFunc",
             [](const ContSet &S, const Eigen::VectorXd &d) { return scalar(S.supportFunc(column(d))); },
             py::arg("d"))
        .def("randPoint",
             [](const ContSet &S, int64_t N, cora::Rng &rng) { return matrix(S.randPoint(N, rng)); },
             py::arg("N"), py::arg("rng"), "N random points as columns (..., n, N)");
}

void bindInterval(py::module_ &m) {
    py::class_<Interval, ContSet>(m, "Interval", "A box, or a matrix of intervals: inf <= x <= sup.")
        .def(py::init([](const torch::Tensor &inf, const torch::Tensor &sup) {
                 return Interval(column(inf), column(sup));
             }),
             py::arg("inf"), py::arg("sup"))
        .def(py::init([](const Eigen::VectorXd &inf, const Eigen::VectorXd &sup) {
                 return Interval(column(inf), column(sup));
             }),
             py::arg("inf"), py::arg("sup"))
        .def_static("matrix",
                    [](const torch::Tensor &inf, const torch::Tensor &sup) {
                        return Interval(fromTorch(inf), fromTorch(sup));
                    },
                    py::arg("inf"), py::arg("sup"),
                    "An interval matrix: bounds (..., n, n) are taken as they are, not as vectors")
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
        .def("mtimes", [](const Interval &I, const torch::Tensor &M) { return I.mtimes(fromTorch(M)); },
             py::arg("M"))
        .def("mtimes", [](const Interval &I, const Eigen::MatrixXd &M) { return I.mtimes(fromEigen(M)); },
             py::arg("M"))
        .def("plus", &Interval::plus, py::arg("I2"))
        .def("contains", [](const Interval &I, const torch::Tensor &p) { return I.contains(column(p)); },
             py::arg("p"))
        .def("contains", [](const Interval &I, const Eigen::VectorXd &p) { return I.contains(column(p)); },
             py::arg("p"));
}

void bindZonotope(py::module_ &m) {
    py::class_<Zonotope, ContSet>(m, "Zonotope", "The zonotope {c + G b : |b|_inf <= 1}.")
        .def(py::init([](const torch::Tensor &c, const torch::Tensor &G) {
                 return Zonotope(column(c), fromTorch(G));
             }),
             py::arg("c"), py::arg("G"), "center c (..., n) and generators G (..., n, m)")
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
        .def("mtimes", [](const Zonotope &Z, const torch::Tensor &M) { return Z.mtimes(fromTorch(M)); },
             py::arg("M"), "M * Z for a matrix M (..., n, n)")
        .def("mtimes", [](const Zonotope &Z, const Eigen::MatrixXd &M) { return Z.mtimes(fromEigen(M)); },
             py::arg("M"))
        .def("mtimes", [](const Zonotope &Z, const Interval &M) { return Z.mtimes(M); }, py::arg("M"),
             "[M] * Z for an interval matrix")
        .def("plus", &Zonotope::plus, py::arg("Z2"))
        .def("linComb", &Zonotope::linComb, py::arg("Z2"));
}

void bindLinearSys(py::module_ &m) {
    py::class_<Reach>(m, "Reach", "The reachable sets: lists of zonotopes.")
        .def_readonly("timeInt", &Reach::timeInt, "timeInt[k] over [k*timeStep, (k+1)*timeStep]")
        .def_readonly("timePoint", &Reach::timePoint, "timePoint[k] at k*timeStep");

    py::class_<LinearSys>(m, "LinearSys", "The linear system x' = A x.")
        .def(py::init([](const torch::Tensor &A, bool customBackward) {
                 return LinearSys(fromTorch(A, customBackward));
             }),
             py::arg("A"), py::arg("customBackward") = false,
             "A (..., n, n); customBackward: hand-written backward pass for the matrix exponential")
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
        .def(
            "simulate",
            [](const LinearSys &sys, const torch::Tensor &x0, double timeStep, double tFinal) {
                return stack(sys.simulate(fromTorch(x0), timeStep, tFinal));
            },
            py::arg("x0"), py::arg("timeStep"), py::arg("tFinal"),
            "Trajectories from the points x0 (..., n, N): (steps + 1, ..., n, N), time first")
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

void bindSpecification(py::module_ &m) {
    py::class_<Specification>(m, "Specification",
                              "Halfspaces {x | a.x <= b} a set must stay in (safeSet) or must not touch "
                              "(unsafeSet).")
        .def_static("safeSet",
                    [](const torch::Tensor &a, double b) { return Specification::safeSet(column(a), b); },
                    py::arg("a"), py::arg("b"))
        .def_static("safeSet",
                    [](const Eigen::VectorXd &a, double b) { return Specification::safeSet(column(a), b); },
                    py::arg("a"), py::arg("b"))
        .def_static("unsafeSet",
                    [](const torch::Tensor &a, double b) { return Specification::unsafeSet(column(a), b); },
                    py::arg("a"), py::arg("b"))
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
    bindSpecification(m);
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
