// test_nonlinearSys_gradient_torch - gradients through reach and simulate with respect to the
// initial set, against central differences

#include "contDynamics/nonlinearSys/nonlinearSys.h"
#include "tensor/torch.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

NonlinearSys vanDerPol() {
    return NonlinearSys(
        [](const std::vector<Expr> &x) {
            return std::vector<Expr>{x[1], (1 - x[0] * x[0]) * x[1] - x[0]};
        },
        2);
}

/// The support along d of every set of the reachable tube, summed.
torch::Tensor reachLoss(const torch::Tensor &c, const torch::Tensor &G, const torch::Tensor &d) {
    const Reach R = vanDerPol().reach(Zonotope(fromTorch(c.unsqueeze(-1)), fromTorch(G)), 0.02, 0.2, 4);
    const Tensor direction = fromTorch(d.unsqueeze(-1));
    torch::Tensor total = torch::zeros({}, c.options());
    for (const Zonotope &Z : R.timePoint) total = total + toTorch(Z.supportFunc(direction)).sum();
    for (const Zonotope &Z : R.timeInt) total = total + toTorch(Z.supportFunc(direction)).sum();
    return total;
}

/// The first state at the end of a simulation of the start points x0 (2, N), summed.
torch::Tensor simulateLoss(const torch::Tensor &x0) {
    return toTorch(vanDerPol().simulate(fromTorch(x0), 0.05, 0.5).back()).sum();
}

/// The largest relative difference between the autograd gradient of f in x0 and central differences.
template <class F> double gradient_error(F f, const torch::Tensor &x0) {
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
    const double cError = gradient_error([&](const torch::Tensor &c) { return reachLoss(c, G0, d); }, c0);
    const double gError = gradient_error([&](const torch::Tensor &G) { return reachLoss(c0, G, d); }, G0);
    check(cError < 1e-5, "reach: the gradient in the center is off by " + std::to_string(cError));
    check(gError < 1e-5, "reach: the gradient in the generators is off by " + std::to_string(gError));
}

void simulate_is_differentiable() {
    const torch::TensorOptions opts = torch::TensorOptions().dtype(torch::kDouble);
    const torch::Tensor x0 = torch::tensor({{1.4, 0.5}, {2.3, -1.0}}, opts);
    const double error = gradient_error(simulateLoss, x0);
    check(error < 1e-6, "simulate: the gradient is off by " + std::to_string(error));
}

} // namespace

int main() {
    setBackend("torch");
    reach_is_differentiable();
    simulate_is_differentiable();
    return test::finish("nonlinearSys gradient");
}
