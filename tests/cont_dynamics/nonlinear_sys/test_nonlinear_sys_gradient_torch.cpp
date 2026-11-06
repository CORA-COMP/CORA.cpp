// test_nonlinear_sys_gradient_torch - gradients through reach and simulate with respect to the
//
// initial set, against central differences
//
// CORA: contDynamics/@contDynamics/reach.m
// CORA: contDynamics/@nonlinearSys/simulate.m
// CORA authors: Matthias Althoff, Niklas Kochdumper
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_dynamics/nonlinear_sys/nonlinear_sys.h"
#include "cora/global/backend/torch.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

NonlinearSys van_der_pol() {
    return NonlinearSys(
        [](const std::vector<Expr> &x) {
            return std::vector<Expr>{x[1], (1 - x[0] * x[0]) * x[1] - x[0]};
        },
        2);
}

/// The support along d of every set of the reachable tube, summed.
torch::Tensor reach_loss(const torch::Tensor &c, const torch::Tensor &G, const torch::Tensor &d) {
    const ReachSet R = van_der_pol().reach(
        {.initial_set = Zonotope(from_torch(c.unsqueeze(-1)), from_torch(G)), .t_final = 0.2},
        alg::Linearization{.time_step = 0.02, .taylor_terms = 4});
    const Tensor direction = from_torch(d.unsqueeze(-1));
    torch::Tensor total = torch::zeros({}, c.options());
    for (const Zonotope &Z : R.time_point)
        total = total + to_torch(Z.support_function(direction)).sum();
    for (const Zonotope &Z : R.time_interval)
        total = total + to_torch(Z.support_function(direction)).sum();
    return total;
}

/// The first state at the end of a simulation of the start points x0 (2, N), summed.
torch::Tensor simulate_loss(const torch::Tensor &x0) {
    return to_torch(van_der_pol()
                        .simulate({.initial_state = from_torch(x0), .t_final = 0.5},
                                  alg::RungeKutta4{.time_step = 0.05})
                        .back())
        .sum();
}

/// The largest relative difference between the autograd gradient of f in x0 and central
/// differences.
template <class F>
double gradient_error(F f, const torch::Tensor &x0) {
    torch::Tensor x = x0.clone().requires_grad_(true);
    const torch::Tensor grad = torch::autograd::grad({f(x)}, {x})[0].detach();
    const double h = 1e-6;
    double worst = 0.0;
    for (int64_t at = 0; at < x0.numel(); ++at) {
        auto value = [&](double shift) {
            torch::NoGradGuard no_grad;
            torch::Tensor y = x0.clone();
            y.view({-1})[at] += shift;
            return f(y).template item<double>();
        };
        const double want = (value(h) - value(-h)) / (2 * h);
        const double got = grad.reshape({-1})[at].item<double>();
        worst = std::max(worst, std::abs(want - got) / (1.0 + std::abs(want)));
    }
    return worst;
}

void reach_is_differentiable() {
    const torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kDouble);
    const torch::Tensor d = torch::tensor({0.6, -0.8}, opts);
    const torch::Tensor c0 = torch::tensor({1.4, 2.3}, opts);
    const torch::Tensor G0 = torch::tensor({{0.05, 0.01}, {-0.02, 0.06}}, opts);
    const double c_error =
        gradient_error([&](const torch::Tensor &c) { return reach_loss(c, G0, d); }, c0);
    const double g_error =
        gradient_error([&](const torch::Tensor &G) { return reach_loss(c0, G, d); }, G0);
    check(c_error < 1e-5, "reach: the gradient in the center is off by " + std::to_string(c_error));
    check(g_error < 1e-5,
          "reach: the gradient in the generators is off by " + std::to_string(g_error));
}

void simulate_is_differentiable() {
    const torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kDouble);
    const torch::Tensor x0 = torch::tensor({{1.4, 0.5}, {2.3, -1.0}}, opts);
    const double error = gradient_error(simulate_loss, x0);
    check(error < 1e-6, "simulate: the gradient is off by " + std::to_string(error));
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    const BackendScope backend_scope("torch");
    reach_is_differentiable();
    simulate_is_differentiable();
    return test::finish("nonlinearSys gradient");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
