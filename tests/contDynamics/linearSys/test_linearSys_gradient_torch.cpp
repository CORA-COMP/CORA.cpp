// test_linearSys_gradient_torch - gradients through the whole reachability computation, with
// autograd and with the hand-written backward pass of the matrix exponential, against central
// differences.

#include "contDynamics/linearSys/linearSys.h"
#include "specification/specification.h"
#include "tensor/torch.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

const Algorithm kAlgorithms[] = {Algorithm::Standard, Algorithm::WrappingFree};

std::string name(Algorithm a) { return a == Algorithm::Standard ? "standard" : "wrapping-free"; }

double max_diff(const torch::Tensor &a, const torch::Tensor &b) {
    return (a.to(torch::kCPU) - b.to(torch::kCPU)).abs().max().item<double>();
}

/// A scalar of the reachable tube: the support of every step's enclosure along `d`.
torch::Tensor loss(const torch::Tensor &A, const torch::Tensor &c, const torch::Tensor &G,
                   Algorithm algorithm, bool customBackward, const torch::Tensor &d) {
    const Zonotope X0(fromTorch(c.unsqueeze(-1), customBackward), fromTorch(G, customBackward));
    const Reach r = LinearSys(fromTorch(A, customBackward)).reach(X0, 0.1, 1.0, 6, algorithm);
    const Tensor direction = fromTorch(d.unsqueeze(-1), customBackward);
    torch::Tensor total = torch::zeros({}, A.options());
    for (const Zonotope &Z : r.timeInt) total = total + toTorch(Z.supportFunc(direction)).sum();
    return total;
}

void gradients(const torch::TensorOptions &opts, const std::string &where) {
    const int64_t n = 3, m = 3;
    const torch::Tensor d = torch::randn({n}, opts);
    for (const Algorithm algorithm : kAlgorithms) {
        const std::string what = where + ": " + name(algorithm);
        const torch::Tensor A0 = torch::randn({n, n}, opts) * 0.5;
        const torch::Tensor c0 = torch::randn({n}, opts), G0 = torch::randn({n, m}, opts);

        std::vector<torch::Tensor> grads[2];
        for (const bool custom : {false, true}) {
            torch::Tensor A = A0.clone().requires_grad_(true), c = c0.clone().requires_grad_(true),
                          G = G0.clone().requires_grad_(true);
            grads[custom] = torch::autograd::grad({loss(A, c, G, algorithm, custom, d)}, {A, c, G});
        }
        for (int i = 0; i < 3; ++i)
            check(max_diff(grads[0][i], grads[1][i]) < 1e-9,
                  what + ": the custom backward differs from autograd (input " + std::to_string(i) +
                      ")");

        // Central differences over every entry of A, c and G.
        const double h = 1e-6;
        const torch::Tensor tensors[3] = {A0, c0, G0};
        double worst = 0.0;
        for (int which = 0; which < 3; ++which)
            for (int64_t at = 0; at < tensors[which].numel(); ++at) {
                auto value = [&](double shift) {
                    torch::NoGradGuard no_grad;
                    torch::Tensor in[3] = {A0.clone(), c0.clone(), G0.clone()};
                    in[which].view({-1})[at] += shift;
                    return loss(in[0], in[1], in[2], algorithm, false, d).item<double>();
                };
                const double want = (value(h) - value(-h)) / (2 * h);
                const double got = grads[0][which].to(torch::kCPU).view({-1})[at].item<double>();
                worst = std::max(worst, std::abs(want - got) / (1.0 + std::abs(want)));
            }
        check(worst < 1e-5,
              what + ": a gradient is off from central differences by " + std::to_string(worst));
    }
}

/// The margin of a specification is differentiable too: how far the enclosure sits from the
/// wall, with respect to the system, moves the way a step in that direction says.
void the_margin_to_a_wall_has_a_gradient(const torch::TensorOptions &opts,
                                         const std::string &where) {
    const torch::Tensor A0 = torch::tensor({{-0.2, 1.0}, {-1.0, -0.2}}, opts);
    const Halfspace wall{fromTorch(torch::tensor({{1.0}, {0.0}}, opts)), 2.0};
    auto margin = [&](const torch::Tensor &A) {
        const Zonotope X0(fromTorch(torch::tensor({{1.0}, {0.5}}, opts)),
                          fromTorch(torch::tensor({{0.1, 0.0}, {0.0, 0.2}}, opts)));
        const Reach r = LinearSys(fromTorch(A)).reach(X0, 0.1, 0.5, 8);
        // Distance of the last enclosure to the wall x1 <= 2.
        return wall.b - toTorch(r.timeInt.back().supportFunc(wall.a)).sum();
    };
    torch::Tensor A = A0.clone().requires_grad_(true);
    const torch::Tensor g = torch::autograd::grad({margin(A)}, {A})[0];
    const double h = 1e-6;
    torch::Tensor up = A0.clone(), down = A0.clone();
    up[0][1] += h;
    down[0][1] -= h;
    torch::NoGradGuard no_grad;
    const double want = (margin(up).item<double>() - margin(down).item<double>()) / (2 * h);
    check(std::abs(g[0][1].item<double>() - want) < 1e-6, where + ": d margin / dA01");
    check(g.abs().sum().item<double>() > 0.0, where + ": the margin depends on A");
}

} // namespace

int main() {
    std::vector<torch::TensorOptions> all{torch::TensorOptions().dtype(torch::kDouble)};
    std::vector<std::string> names{"cpu"};
    if (torch::cuda::is_available()) {
        all.push_back(torch::TensorOptions().dtype(torch::kDouble).device(torch::kCUDA, 0));
        names.push_back("gpu");
    }
    setBackend("torch");
    torch::manual_seed(7);
    for (std::size_t i = 0; i < all.size(); ++i) {
        gradients(all[i], names[i]);
        the_margin_to_a_wall_has_a_gradient(all[i], names[i]);
    }
    return test::finish("linearSys gradient");
}
