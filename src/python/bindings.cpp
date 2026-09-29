// Python bindings: `coracpp.reach`, the reachable sets of a linear system.
//
// Torch tensors run on the libtorch backend — batched over sets and systems, on any
// device, differentiable with autograd — and numpy arrays on the Eigen backend, so the
// same call picks the backend by what it is given. Both go through the one algorithm in
// contDynamics/linear_sys.cpp.
//
//     import torch, coracpp
//     r = coracpp.reach(A, c, G, time_step=0.1, t_final=2.0)   # A (..., n, n), c (..., n),
//     r.time_int_G                                              # G (..., n, m)
//
// Results stack the steps in a leading dimension: `time_int_*` has `steps` entries,
// `time_point_*` has `steps + 1`.

#include "contDynamics/linear_sys.h"
#include "tensor/eigen.h"
#include "tensor/torch.h"

#include <pybind11/eigen.h>
#include <torch/extension.h>

namespace py = pybind11;
using namespace cora::ct;

namespace {

struct PyReach {
    py::object time_int_c, time_int_G, time_point_c, time_point_G;
};

Algorithm parse(const std::string &algorithm) {
    if (algorithm == "standard") return Algorithm::Standard;
    if (algorithm == "wrapping-free") return Algorithm::WrappingFree;
    throw py::value_error("algorithm must be 'standard' or 'wrapping-free'");
}

/// The steps as one tensor with the step first. Steps of a run can differ in batch shape
/// (an unbatched set under a batched system starts unbatched), so they broadcast first.
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
            py::cast(stack_torch(r.time_point, true)), py::cast(stack_torch(r.time_point, false))};
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
            stack_numpy(r.time_point, true), stack_numpy(r.time_point, false)};
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
}
