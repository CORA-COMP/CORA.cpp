// Python bindings: `coracpp.reach`, the reachable sets of a linear system,
// `coracpp.simulate` and `coracpp.rand_point` for trajectories from random points of the
// initial set, and `coracpp.Specification`, a halfspace they must stay in or out of.
//
// Torch tensors run on the libtorch backend — batched over sets and systems, on any
// device, differentiable with autograd — and numpy arrays on the Eigen backend, so the
// same call picks the backend by what it is given. Both go through the one algorithm in
// contDynamics/linearSys/linearSys.cpp.
//
//     import torch, coracpp
//     r = coracpp.reach(A, c, G, time_step=0.1, t_final=2.0)   # A (..., n, n), c (..., n),
//     r.time_int_G                                              # G (..., n, m)
//
// Results stack the steps in a leading dimension: `time_int_*` has `steps` entries,
// `time_point_*` has `steps + 1`.

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
    py::object time_int_c, time_int_G, time_point_c, time_point_G;
    std::shared_ptr<Reach> sets;
};

Algorithm parse(const std::string &algorithm) {
    if (algorithm == "standard") return Algorithm::Standard;
    if (algorithm == "wrapping-free") return Algorithm::WrappingFree;
    throw py::value_error("algorithm must be 'standard' or 'wrapping-free'");
}

/// The steps as one tensor with the step first. Steps of a run can differ in batch shape (an
/// enclosure of a batched system next to a set that is not), so they broadcast first.
torch::Tensor stack_torch(const std::vector<Zonotope> &sets, bool centre) {
    std::vector<torch::Tensor> parts;
    for (const Zonotope &Z : sets) {
        const torch::Tensor t = to_torch(centre ? Z.c : Z.G);
        parts.push_back(centre ? t.squeeze(-1) : t);
    }
    return torch::stack(torch::broadcast_tensors(parts), 0);
}

PyReach reach_torch(const torch::Tensor &A, const torch::Tensor &c, const torch::Tensor &G,
                    double time_step, double t_final, int taylor_terms,
                    const std::string &algorithm, bool custom_backward) {
    const Zonotope X0{from_torch(c.unsqueeze(-1), custom_backward), from_torch(G, custom_backward)};
    const Reach r = LinearSys(from_torch(A, custom_backward))
                        .reach(X0, time_step, t_final, taylor_terms, parse(algorithm));
    return {py::cast(stack_torch(r.time_int, true)), py::cast(stack_torch(r.time_int, false)),
            py::cast(stack_torch(r.time_point, true)), py::cast(stack_torch(r.time_point, false)),
            std::make_shared<Reach>(r)};
}

py::object stack_numpy(const std::vector<Zonotope> &sets, bool centre) {
    py::list parts;
    for (const Zonotope &Z : sets) {
        Eigen::MatrixXd M = to_eigen(centre ? Z.c : Z.G);
        if (centre)
            parts.append(py::cast(Eigen::VectorXd(M.col(0))));
        else
            parts.append(py::cast(M));
    }
    return py::module_::import("numpy").attr("stack")(parts);
}

PyReach reach_numpy(const Eigen::MatrixXd &A, const Eigen::VectorXd &c, const Eigen::MatrixXd &G,
                    double time_step, double t_final, int taylor_terms,
                    const std::string &algorithm) {
    const Zonotope X0{from_eigen(c), from_eigen(G)};
    const Reach r =
        LinearSys(from_eigen(A)).reach(X0, time_step, t_final, taylor_terms, parse(algorithm));
    return {stack_numpy(r.time_int, true), stack_numpy(r.time_int, false),
            stack_numpy(r.time_point, true), stack_numpy(r.time_point, false),
            std::make_shared<Reach>(r)};
}

/// A specification made from torch tensors or numpy arrays lives on that backend, and is
/// checked against reachable sets from the same one.
Specification safe_torch(const torch::Tensor &a, double b) {
    return Specification::safe_set(from_torch(a.unsqueeze(-1)), b);
}
Specification unsafe_torch(const torch::Tensor &a, double b) {
    return Specification::unsafe_set(from_torch(a.unsqueeze(-1)), b);
}
Specification safe_numpy(const Eigen::VectorXd &a, double b) {
    return Specification::safe_set(from_eigen(a), b);
}
Specification unsafe_numpy(const Eigen::VectorXd &a, double b) {
    return Specification::unsafe_set(from_eigen(a), b);
}

/// `N` random points of the zonotope (c, G), columns `(..., n, N)`. The seed makes a call
/// repeatable; give a different one for a different draw.
torch::Tensor rand_point_torch(const torch::Tensor &c, const torch::Tensor &G, int64_t N,
                               uint64_t seed, bool extreme) {
    cora::Rng rng(seed);
    const Zonotope Z(from_torch(c.unsqueeze(-1)), from_torch(G));
    return to_torch(Z.rand_point(N, rng, extreme));
}

Eigen::MatrixXd rand_point_numpy(const Eigen::VectorXd &c, const Eigen::MatrixXd &G, int64_t N,
                                 uint64_t seed, bool extreme) {
    cora::Rng rng(seed);
    return to_eigen(Zonotope(from_eigen(c), from_eigen(G)).rand_point(N, rng, extreme));
}

/// The trajectories from the points `x0` `(..., n, N)`, the time points stacked first.
torch::Tensor simulate_torch(const torch::Tensor &A, const torch::Tensor &x0, double time_step,
                             double t_final) {
    std::vector<torch::Tensor> points;
    for (const Tensor &x : LinearSys(from_torch(A)).simulate(from_torch(x0), time_step, t_final))
        points.push_back(to_torch(x));
    return torch::stack(torch::broadcast_tensors(points), 0);
}

py::object simulate_numpy(const Eigen::MatrixXd &A, const Eigen::MatrixXd &x0, double time_step,
                          double t_final) {
    py::list points;
    for (const Tensor &x : LinearSys(from_eigen(A)).simulate(from_eigen(x0), time_step, t_final))
        points.append(py::cast(to_eigen(x)));
    return py::module_::import("numpy").attr("stack")(points);
}

} // namespace

PYBIND11_MODULE(coracpp, m) {
    m.doc() = "CORA.cpp: reachability of linear systems on Eigen (numpy) or libtorch (torch)";

    py::class_<PyReach>(m, "Reach")
        .def_readonly("time_int_c", &PyReach::time_int_c)
        .def_readonly("time_int_G", &PyReach::time_int_G)
        .def_readonly("time_point_c", &PyReach::time_point_c)
        .def_readonly("time_point_G", &PyReach::time_point_G);

    m.def("reach", &reach_torch, py::arg("A"), py::arg("c"), py::arg("G"), py::arg("time_step"),
          py::arg("t_final"), py::arg("taylor_terms") = 10, py::arg("algorithm") = "standard",
          py::arg("custom_backward") = false,
          "Reachable sets of x' = A x from the zonotope (c, G), on libtorch. `A` (..., n, n), "
          "`c` (..., n), `G` (..., n, m); leading dimensions broadcast. `algorithm` is "
          "'standard' or 'wrapping-free'; `custom_backward` uses the hand-written backward "
          "pass of the matrix exponential.");
    m.def("reach", &reach_numpy, py::arg("A"), py::arg("c"), py::arg("G"), py::arg("time_step"),
          py::arg("t_final"), py::arg("taylor_terms") = 10, py::arg("algorithm") = "standard",
          "The same for numpy arrays, on Eigen: one set, `A` (n, n), `c` (n,), `G` (n, m).");

    m.def("rand_point", &rand_point_torch, py::arg("c"), py::arg("G"), py::arg("N"),
          py::arg("seed") = 0, py::arg("extreme") = false,
          "N random points c + G b of the zonotope (c, G), on libtorch: `c` (..., n), `G` "
          "(..., n, m), the result (..., n, N). With `extreme`, b is a corner of the cube.");
    m.def("rand_point", &rand_point_numpy, py::arg("c"), py::arg("G"), py::arg("N"),
          py::arg("seed") = 0, py::arg("extreme") = false,
          "The same for numpy arrays, on Eigen: `c` (n,), `G` (n, m), the result (n, N).");
    m.def("simulate", &simulate_torch, py::arg("A"), py::arg("x0"), py::arg("time_step"),
          py::arg("t_final"),
          "Trajectories of x' = A x from the points `x0` (..., n, N), exact through e^{A t}, on "
          "libtorch. The result is (steps + 1, ..., n, N): the time points first.");
    m.def("simulate", &simulate_numpy, py::arg("A"), py::arg("x0"), py::arg("time_step"),
          py::arg("t_final"), "The same for numpy arrays, on Eigen: `x0` (n, N).");

    py::class_<Specification>(m, "Specification",
                              "A halfspace {x | a.x <= b} the reachable set must stay in "
                              "(safe_set) or must not touch (unsafe_set).")
        .def_static("safe_set", &safe_torch, py::arg("a"), py::arg("b"))
        .def_static("safe_set", &safe_numpy, py::arg("a"), py::arg("b"))
        .def_static("unsafe_set", &unsafe_torch, py::arg("a"), py::arg("b"))
        .def_static("unsafe_set", &unsafe_numpy, py::arg("a"), py::arg("b"))
        .def(
            "check",
            [](const Specification &spec, const PyReach &r) { return spec.check(r.sets->time_int); },
            py::arg("reach"), "Whether every time-interval set satisfies the specification.")
        .def(
            "first_violation",
            [](const Specification &spec, const PyReach &r) {
                return spec.first_violation(r.sets->time_int);
            },
            py::arg("reach"), "The first violating step, or -1.");
}
