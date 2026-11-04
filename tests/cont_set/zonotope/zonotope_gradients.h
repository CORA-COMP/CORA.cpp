// zonotope_gradients - what the torch tests of the zonotope operations share: a gradient check
//
// Syntax:   gradcheck("abs", [](const Inputs &x) { return Outputs{Zonotope(x[0], x[1]).abs()...};
// },
//                     {center, generators});
// Checks:   the gradient that autograd gives through the torch backend, of a fixed random weighting
//           of all outputs, equals central differences (step 1e-6, float64); the inputs are chosen
//           away from the kinks of the piecewise operations
// See also: tests/global/tensor/test_tensor_gradcheck_torch.cpp

#pragma once

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/backend/torch.h"
#include "testing.h"

#include <functional>

namespace test {

using Inputs = std::vector<cora::Tensor>;
using Function = std::function<Inputs(const Inputs &)>;

/// sum(w * f(x)) over all outputs, with weights drawn on first use.
inline torch::Tensor weighted_sum(const Function &f, const std::vector<torch::Tensor> &x,
                                  std::vector<torch::Tensor> &weights) {
    Inputs in;
    for (const torch::Tensor &t : x) in.push_back(cora::from_torch(t));
    const Inputs out = f(in);
    torch::Tensor loss = torch::zeros({}, torch::kDouble);
    for (std::size_t i = 0; i < out.size(); ++i) {
        const torch::Tensor y = cora::to_torch(out[i]);
        if (weights.size() <= i) weights.push_back(torch::randn(y.sizes(), torch::kDouble));
        loss = loss + (y * weights[i]).sum();
    }
    return loss;
}

/// Autograd's gradient of f at the inputs against central differences.
inline void gradcheck(const std::string &name, const Function &f,
                      const std::vector<torch::Tensor> &inputs) {
    std::vector<torch::Tensor> x, weights;
    for (const torch::Tensor &t : inputs) x.push_back(t.detach().clone().requires_grad_(true));
    const torch::Tensor loss = weighted_sum(f, x, weights);
    const std::vector<torch::Tensor> grads =
        torch::autograd::grad({loss}, x, {}, false, false, true);
    const double h = 1e-6;
    bool ok = true;
    for (std::size_t k = 0; k < x.size() && ok; ++k) {
        const torch::Tensor flat = x[k].detach().flatten();
        for (int64_t j = 0; j < flat.numel() && ok; ++j) {
            std::vector<torch::Tensor> plus, minus;
            for (const torch::Tensor &t : x) {
                plus.push_back(t.detach().clone());
                minus.push_back(t.detach().clone());
            }
            plus[k].view(-1)[j] += h;
            minus[k].view(-1)[j] -= h;
            const double numeric = (weighted_sum(f, plus, weights).item<double>() -
                                    weighted_sum(f, minus, weights).item<double>()) /
                                   (2 * h);
            const double analytic = grads[k].defined() ? grads[k].flatten()[j].item<double>() : 0.0;
            if (std::abs(numeric - analytic) > 1e-5 + 1e-4 * std::abs(numeric)) {
                std::cerr << "  " << name << ": input " << k << " entry " << j << ": autograd "
                          << analytic << ", differences " << numeric << "\n";
                ok = false;
            }
        }
    }
    check(ok, name + ": autograd matches central differences");
}

/// A center (n, 1) and generators (n, m) with entries of magnitude in [0.3, 2], away from zero.
inline std::vector<torch::Tensor> random_zonotope(int64_t n, int64_t m) {
    const auto away = [](std::vector<int64_t> shape) {
        const torch::Tensor sign = torch::randint(0, 2, shape, torch::kDouble) * 2 - 1;
        return sign * (0.3 + 1.7 * torch::rand(shape, torch::kDouble));
    };
    return {away({n, 1}), away({n, m})};
}

/// The center and generators of a zonotope, as the outputs of a function.
inline Inputs parts(const cora::Zonotope &Z) { return {Z.center(), Z.generators()}; }

} // namespace test
