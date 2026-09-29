// bindings - the compiled half of the `coracpp` Python package (`_coracpp`)
//
// The plotting and the colors are Python, next to this file in coracpp/. Torch tensors run on
// the libtorch backend (batched, on any device, differentiable), numpy arrays on Eigen, so a
// call picks its backend by the type of its arguments. Names and arguments follow CORA.
//
//     R = coracpp.reach(A, c, G, timeStep=0.1, tFinal=2.0)     # A (..., n, n), c (..., n),
//     R.timeInt_G                                                # G (..., n, m)
//
// Results stack the steps in a leading dimension: timeInt_* has `steps` entries, timePoint_*
// has `steps + 1`.

#include "contDynamics/linearSys/linearSys.h"
#include "global/rng.h"
#include "specification/specification.h"
#include "tensor/eigen.h"
#include "tensor/torch.h"

#include <pybind11/eigen.h>
#include <torch/extension.h>

namespace py = pybind11;
using namespace cora::ct;

namespace {

/// The reachable sets, stacked per step for Python and kept as they are for specifications.
struct PyReach {
    py::object timeInt_c, timeInt_G, timePoint_c, timePoint_G;
    std::shared_ptr<Reach> sets;
};

/// CORA's linAlg: "standard" or "wrapping-free".
Algorithm parseLinAlg(const std::string &linAlg) {
    if (linAlg == "standard") return Algorithm::Standard;
    if (linAlg == "wrapping-free") return Algorithm::WrappingFree;
    throw py::value_error("linAlg must be 'standard' or 'wrapping-free'");
}

// ------------------------------- libtorch ----------------------------------------

/// The steps as one tensor, the step first; they broadcast first since the batch shape of a
/// set can differ from that of its neighbours.
torch::Tensor stackTorch(const std::vector<Zonotope> &sets, bool center) {
    std::vector<torch::Tensor> parts;
    for (const Zonotope &Z : sets) {
        const torch::Tensor t = toTorch(center ? Z.c : Z.G);
        parts.push_back(center ? t.squeeze(-1) : t);
    }
    return torch::stack(torch::broadcast_tensors(parts), 0);
}

PyReach reachTorch(const torch::Tensor &A, const torch::Tensor &c, const torch::Tensor &G,
                   double timeStep, double tFinal, int taylorTerms, const std::string &linAlg,
                   bool customBackward) {
    const Zonotope X0{fromTorch(c.unsqueeze(-1), customBackward), fromTorch(G, customBackward)};
    const Reach R = LinearSys(fromTorch(A, customBackward))
                        .reach(X0, timeStep, tFinal, taylorTerms, parseLinAlg(linAlg));
    return {py::cast(stackTorch(R.timeInt, true)), py::cast(stackTorch(R.timeInt, false)),
            py::cast(stackTorch(R.timePoint, true)), py::cast(stackTorch(R.timePoint, false)),
            std::make_shared<Reach>(R)};
}

torch::Tensor randPointTorch(const torch::Tensor &c, const torch::Tensor &G, int64_t N,
                             uint64_t seed, const std::string &type) {
    cora::Rng rng(seed);
    const Zonotope Z(fromTorch(c.unsqueeze(-1)), fromTorch(G));
    return toTorch(Z.randPoint(N, rng, type));
}

torch::Tensor simulateTorch(const torch::Tensor &A, const torch::Tensor &x0, double timeStep,
                            double tFinal) {
    std::vector<torch::Tensor> points;
    for (const Tensor &x : LinearSys(fromTorch(A)).simulate(fromTorch(x0), timeStep, tFinal))
        points.push_back(toTorch(x));
    return torch::stack(torch::broadcast_tensors(points), 0);
}

torch::Tensor simulateRandomTorch(const torch::Tensor &A, const torch::Tensor &c,
                                  const torch::Tensor &G, int64_t N, double timeStep, double tFinal,
                                  uint64_t seed, const std::string &type) {
    return simulateTorch(A, randPointTorch(c, G, N, seed, type), timeStep, tFinal);
}

Specification safeSetTorch(const torch::Tensor &a, double b) {
    return Specification::safeSet(fromTorch(a.unsqueeze(-1)), b);
}

Specification unsafeSetTorch(const torch::Tensor &a, double b) {
    return Specification::unsafeSet(fromTorch(a.unsqueeze(-1)), b);
}

// -------------------------------- Eigen ------------------------------------------

py::object stackNumpy(const std::vector<Zonotope> &sets, bool center) {
    py::list parts;
    for (const Zonotope &Z : sets) {
        const Eigen::MatrixXd M = toEigen(center ? Z.c : Z.G);
        parts.append(center ? py::cast(Eigen::VectorXd(M.col(0))) : py::cast(M));
    }
    return py::module_::import("numpy").attr("stack")(parts);
}

PyReach reachNumpy(const Eigen::MatrixXd &A, const Eigen::VectorXd &c, const Eigen::MatrixXd &G,
                   double timeStep, double tFinal, int taylorTerms, const std::string &linAlg) {
    const Zonotope X0{fromEigen(c), fromEigen(G)};
    const Reach R = LinearSys(fromEigen(A)).reach(X0, timeStep, tFinal, taylorTerms,
                                                  parseLinAlg(linAlg));
    return {stackNumpy(R.timeInt, true), stackNumpy(R.timeInt, false),
            stackNumpy(R.timePoint, true), stackNumpy(R.timePoint, false),
            std::make_shared<Reach>(R)};
}

Eigen::MatrixXd randPointNumpy(const Eigen::VectorXd &c, const Eigen::MatrixXd &G, int64_t N,
                               uint64_t seed, const std::string &type) {
    cora::Rng rng(seed);
    return toEigen(Zonotope(fromEigen(c), fromEigen(G)).randPoint(N, rng, type));
}

py::object simulateNumpy(const Eigen::MatrixXd &A, const Eigen::MatrixXd &x0, double timeStep,
                         double tFinal) {
    py::list points;
    for (const Tensor &x : LinearSys(fromEigen(A)).simulate(fromEigen(x0), timeStep, tFinal))
        points.append(py::cast(toEigen(x)));
    return py::module_::import("numpy").attr("stack")(points);
}

py::object simulateRandomNumpy(const Eigen::MatrixXd &A, const Eigen::VectorXd &c,
                               const Eigen::MatrixXd &G, int64_t N, double timeStep, double tFinal,
                               uint64_t seed, const std::string &type) {
    return simulateNumpy(A, randPointNumpy(c, G, N, seed, type), timeStep, tFinal);
}

Specification safeSetNumpy(const Eigen::VectorXd &a, double b) {
    return Specification::safeSet(fromEigen(a), b);
}

Specification unsafeSetNumpy(const Eigen::VectorXd &a, double b) {
    return Specification::unsafeSet(fromEigen(a), b);
}

} // namespace

PYBIND11_MODULE(_coracpp, m) {
    m.doc() = "CORA.cpp: reachability of linear systems on Eigen (numpy) or libtorch (torch)";

    py::class_<PyReach>(m, "Reach", "The reachable sets: timeInt_* (steps), timePoint_* (steps + 1).")
        .def_readonly("timeInt_c", &PyReach::timeInt_c)
        .def_readonly("timeInt_G", &PyReach::timeInt_G)
        .def_readonly("timePoint_c", &PyReach::timePoint_c)
        .def_readonly("timePoint_G", &PyReach::timePoint_G);

    m.def("reach", &reachTorch, py::arg("A"), py::arg("c"), py::arg("G"), py::arg("timeStep"),
          py::arg("tFinal"), py::arg("taylorTerms") = 10, py::arg("linAlg") = "standard",
          py::arg("customBackward") = false,
          "Reachable sets of x' = A x from the zonotope (c, G), on libtorch. A (..., n, n), c "
          "(..., n), G (..., n, m); leading dimensions broadcast. linAlg: 'standard' or "
          "'wrapping-free'. customBackward: hand-written backward pass for the matrix exponential.");
    m.def("reach", &reachNumpy, py::arg("A"), py::arg("c"), py::arg("G"), py::arg("timeStep"),
          py::arg("tFinal"), py::arg("taylorTerms") = 10, py::arg("linAlg") = "standard",
          "The same for numpy arrays, on Eigen: one set, A (n, n), c (n,), G (n, m).");

    m.def("randPoint", &randPointTorch, py::arg("c"), py::arg("G"), py::arg("N"),
          py::arg("seed") = 0, py::arg("type") = "standard",
          "N random points of the zonotope (c, G), on libtorch: c (..., n), G (..., n, m), result "
          "(..., n, N). type: 'standard' (b in [-1, 1]^m) or 'extreme' (b in {-1, 1}^m).");
    m.def("randPoint", &randPointNumpy, py::arg("c"), py::arg("G"), py::arg("N"),
          py::arg("seed") = 0, py::arg("type") = "standard",
          "The same for numpy arrays, on Eigen: c (n,), G (n, m), result (n, N).");

    m.def("simulate", &simulateTorch, py::arg("A"), py::arg("x0"), py::arg("timeStep"),
          py::arg("tFinal"),
          "Trajectories of x' = A x from the points x0 (..., n, N), exact, on libtorch. Result "
          "(steps + 1, ..., n, N): the time points first.");
    m.def("simulate", &simulateNumpy, py::arg("A"), py::arg("x0"), py::arg("timeStep"),
          py::arg("tFinal"), "The same for numpy arrays, on Eigen: x0 (n, N).");
    m.def("simulateRandom", &simulateRandomTorch, py::arg("A"), py::arg("c"), py::arg("G"),
          py::arg("N"), py::arg("timeStep"), py::arg("tFinal"), py::arg("seed") = 0,
          py::arg("type") = "standard", "simulate from N random points of the zonotope (c, G).");
    m.def("simulateRandom", &simulateRandomNumpy, py::arg("A"), py::arg("c"), py::arg("G"),
          py::arg("N"), py::arg("timeStep"), py::arg("tFinal"), py::arg("seed") = 0,
          py::arg("type") = "standard", "The same for numpy arrays, on Eigen.");

    py::class_<Specification>(m, "Specification",
                              "A halfspace {x | a.x <= b} the reachable set must stay in "
                              "(safeSet) or must not touch (unsafeSet).")
        .def_static("safeSet", &safeSetTorch, py::arg("a"), py::arg("b"))
        .def_static("safeSet", &safeSetNumpy, py::arg("a"), py::arg("b"))
        .def_static("unsafeSet", &unsafeSetTorch, py::arg("a"), py::arg("b"))
        .def_static("unsafeSet", &unsafeSetNumpy, py::arg("a"), py::arg("b"))
        .def_property_readonly("type",
                               [](const Specification &spec) {
                                   return spec.type() == SpecType::SafeSet ? "safeSet" : "unsafeSet";
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
            "check",
            [](const Specification &spec, const PyReach &R) { return spec.check(R.sets->timeInt); },
            py::arg("reach"), "Whether every time-interval set satisfies the specification.")
        .def(
            "firstViolation",
            [](const Specification &spec, const PyReach &R) {
                return spec.firstViolation(R.sets->timeInt);
            },
            py::arg("reach"), "The first violating step, or -1.");
}
