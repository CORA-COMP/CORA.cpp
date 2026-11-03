// test_tensor_gradcheck_torch - autograd through every differentiable Tensor operation is right
//
// For each operation tagged differentiable or piecewise in ops.def, the gradient that autograd
// gives through the torch backend equals central differences, in float64, at points away from the
// kinks of the piecewise ones. The conventions test requires a gradcheck("<op>", ...) here for
// each.

#include "cora/global/backend/torch.h"
#include "cora/global/tensor/tensor.h"
#include "testing.h"

#include <functional>

using namespace cora;
using test::check;

namespace {

using Norm = Tensor::Norm;
using Fn = std::function<std::vector<Tensor>(const std::vector<Tensor> &)>;

/// sum(w * f(x)) over all outputs, with fixed weights.
torch::Tensor aux_loss(const Fn &f, const std::vector<torch::Tensor> &x,
                       std::vector<torch::Tensor> &weights) {
    std::vector<Tensor> in;
    for (const torch::Tensor &t : x) in.push_back(from_torch(t));
    const std::vector<Tensor> out = f(in);
    torch::Tensor loss = torch::zeros({}, torch::kDouble);
    for (std::size_t i = 0; i < out.size(); ++i) {
        const torch::Tensor y = to_torch(out[i]);
        if (weights.size() <= i) weights.push_back(torch::randn(y.sizes(), torch::kDouble));
        loss = loss + (y * weights[i]).sum();
    }
    return loss;
}

/// Autograd's gradient of f at the inputs against central differences of step 1e-6.
void gradcheck(const std::string &op, const Fn &f, const std::vector<torch::Tensor> &inputs) {
    std::vector<torch::Tensor> x, weights;
    for (const torch::Tensor &t : inputs) x.push_back(t.detach().clone().requires_grad_(true));
    const torch::Tensor loss = aux_loss(f, x, weights);
    const std::vector<torch::Tensor> grads =
        torch::autograd::grad({loss}, x, {}, false, false, true);
    const double h = 1e-6;
    bool ok = true;
    for (std::size_t k = 0; k < x.size() && ok; ++k) {
        const torch::Tensor flat = x[k].detach().flatten();
        for (int64_t j = 0; j < flat.numel() && ok; ++j) {
            std::vector<torch::Tensor> plus, minus;
            for (std::size_t i = 0; i < x.size(); ++i) {
                plus.push_back(x[i].detach().clone());
                minus.push_back(x[i].detach().clone());
            }
            plus[k].view(-1)[j] += h;
            minus[k].view(-1)[j] -= h;
            const double numeric = (aux_loss(f, plus, weights).item<double>() -
                                    aux_loss(f, minus, weights).item<double>()) /
                                   (2 * h);
            const double analytic = grads[k].defined() ? grads[k].flatten()[j].item<double>() : 0.0;
            if (std::abs(numeric - analytic) > 1e-5 + 1e-4 * std::abs(numeric)) {
                std::cerr << "  " << op << ": input " << k << " entry " << j << ": autograd "
                          << analytic << ", differences " << numeric << "\n";
                ok = false;
            }
        }
    }
    check(ok, op + ": autograd matches central differences");
}

torch::Tensor aux_randn(std::vector<int64_t> shape) { return torch::randn(shape, torch::kDouble); }

/// Entries in (lo, hi), well away from both.
torch::Tensor aux_within(std::vector<int64_t> shape, double lo, double hi) {
    return lo + (hi - lo) * (0.1 + 0.8 * torch::rand(shape, torch::kDouble));
}

/// Entries of magnitude in [0.3, 2], away from zero and from each other in sign changes.
torch::Tensor aux_away(std::vector<int64_t> shape) {
    const torch::Tensor s = torch::randint(0, 2, shape, torch::kDouble) * 2 - 1;
    return s * aux_within(shape, 0.3, 2.0);
}

/// Halfway between integers, so floor, ceil and round are flat around them.
torch::Tensor aux_between(std::vector<int64_t> shape) {
    return torch::floor(aux_randn(shape) * 3) + 0.3 + 0.4 * torch::rand(shape, torch::kDouble);
}

Tensor aux_regular(const Tensor &x) { return x + x.eye_like() * double(x.size(-1)); }
Tensor aux_spd(const Tensor &x) { return x.matmul(x.matrix_transpose()) + x.eye_like() * 4.0; }

void elementwise() {
    using V = std::vector<Tensor>;
    gradcheck("add", [](const V &v) { return V{v[0] + v[1], v[0] + 2.0}; },
              {aux_randn({3, 4}), aux_randn({3, 1})});
    gradcheck("subtract", [](const V &v) { return V{v[0] - v[1], 1.0 - v[0]}; },
              {aux_randn({2, 3, 4}), aux_randn({3, 4})});
    gradcheck("multiply", [](const V &v) { return V{v[0].multiply(v[1])}; },
              {aux_randn({3, 4}), aux_randn({3, 4})});
    gradcheck("divide", [](const V &v) { return V{v[0].divide(v[1])}; },
              {aux_randn({3, 4}), aux_away({3, 4})});
    gradcheck("pow", [](const V &v) { return V{v[0].pow(v[1])}; },
              {aux_within({3, 4}, 0.5, 2), aux_randn({3, 4})});
    gradcheck("maximum", [](const V &v) { return V{v[0].maximum(v[1])}; },
              {aux_randn({3, 4}), aux_randn({3, 4}) + 5});
    gradcheck("minimum", [](const V &v) { return V{v[0].minimum(v[1])}; },
              {aux_randn({3, 4}), aux_randn({3, 4}) + 5});
    gradcheck("remainder", [](const V &v) { return V{v[0].remainder(v[1])}; },
              {aux_within({3, 4}, 0.1, 0.4), aux_within({3, 4}, 0.6, 1)});
    gradcheck("atan2", [](const V &v) { return V{v[0].atan2(v[1])}; },
              {aux_randn({3, 4}), aux_away({3, 4})});
    const auto one = [](Tensor (Tensor::*op)() const) {
        return [op](const V &v) { return V{(v[0].*op)()}; };
    };
    gradcheck("negative", one(&Tensor::negative), {aux_randn({3, 4})});
    gradcheck("abs", [](const V &v) { return V{v[0].abs()}; }, {aux_away({3, 4})});
    gradcheck("sign", one(&Tensor::sign), {aux_away({3, 4})});
    gradcheck("floor", one(&Tensor::floor), {aux_between({3, 4})});
    gradcheck("ceil", one(&Tensor::ceil), {aux_between({3, 4})});
    gradcheck("round", one(&Tensor::round), {aux_between({3, 4}) + 0.5});
    gradcheck("sqrt", one(&Tensor::sqrt), {aux_within({3, 4}, 0.5, 4)});
    gradcheck("exp", one(&Tensor::exp), {aux_randn({3, 4})});
    gradcheck("log", one(&Tensor::log), {aux_within({3, 4}, 0.5, 4)});
    gradcheck("sin", one(&Tensor::sin), {aux_randn({3, 4})});
    gradcheck("cos", one(&Tensor::cos), {aux_randn({3, 4})});
    gradcheck("tan", one(&Tensor::tan), {aux_within({3, 4}, -1, 1)});
    gradcheck("asin", one(&Tensor::asin), {aux_within({3, 4}, -0.8, 0.8)});
    gradcheck("acos", one(&Tensor::acos), {aux_within({3, 4}, -0.8, 0.8)});
    gradcheck("atan", one(&Tensor::atan), {aux_randn({3, 4})});
    gradcheck("sinh", one(&Tensor::sinh), {aux_randn({3, 4})});
    gradcheck("cosh", one(&Tensor::cosh), {aux_randn({3, 4})});
    gradcheck("tanh", one(&Tensor::tanh), {aux_randn({3, 4})});
    gradcheck("asinh", one(&Tensor::asinh), {aux_randn({3, 4})});
    gradcheck("acosh", one(&Tensor::acosh), {aux_within({3, 4}, 1.5, 4)});
    gradcheck("atanh", one(&Tensor::atanh), {aux_within({3, 4}, -0.8, 0.8)});
    gradcheck("square", one(&Tensor::square), {aux_randn({3, 4})});
    gradcheck("sigmoid", one(&Tensor::sigmoid), {aux_randn({3, 4})});
    gradcheck("clip", [](const V &v) { return V{v[0].clip(-1, 1)}; }, {aux_away({3, 4}) * 0.5});
    gradcheck("where",
              [](const V &v) {
                  return V{Tensor::where(v[0].detach().greater(v[1].detach()), v[0], v[1])};
              },
              {aux_randn({3, 4}), aux_randn({3, 4}) + 3});
    gradcheck("to", [](const V &v) { return V{v[0].to("cpu") * 2.0, v[0].to(DType::float64)}; },
              {aux_randn({3, 2})});
}

void reductions() {
    using V = std::vector<Tensor>;
    const torch::Tensor x = aux_randn({2, 3, 4});
    gradcheck("sum", [](const V &v) { return V{v[0].sum(1), v[0].sum(-1, true), v[0].sum()}; },
              {x});
    gradcheck("prod", [](const V &v) { return V{v[0].prod(-1)}; }, {aux_away({2, 3, 4})});
    gradcheck("max", [](const V &v) { return V{v[0].max(1), v[0].max()}; }, {x});
    gradcheck("min", [](const V &v) { return V{v[0].min(-1, true)}; }, {x});
    gradcheck("cumsum", [](const V &v) { return V{v[0].cumsum(1)}; }, {x});
    gradcheck("mean", [](const V &v) { return V{v[0].mean(0)}; }, {x});
    gradcheck("softmax", [](const V &v) { return V{v[0].softmax(-1)}; }, {x});
}

void shapes() {
    using V = std::vector<Tensor>;
    const torch::Tensor x = aux_randn({2, 3, 4});
    gradcheck("reshape", [](const V &v) { return V{v[0].reshape({4, 6})}; }, {x});
    gradcheck("permute_dims", [](const V &v) { return V{v[0].permute_dims({2, 0, 1})}; }, {x});
    gradcheck("broadcast_to", [](const V &v) { return V{v[0].broadcast_to({2, 3, 4})}; },
              {aux_randn({3, 1})});
    gradcheck("slice", [](const V &v) { return V{v[0].slice(-1, 1, 4, 2)}; }, {x});
    gradcheck("take", [](const V &v) { return V{v[0].take({3, 0, 3}, -1)}; }, {x});
    gradcheck("take_along_axis",
              [](const V &v) {
                  return V{v[0].take_along_axis(Tensor::like(v[0], {2, 0}, {1, 1, 2}), -1)};
              },
              {x});
    gradcheck("concat", [](const V &v) { return V{Tensor::concat({v[0], v[1]}, -1)}; },
              {aux_randn({3, 2}), aux_randn({3, 4})});
    gradcheck("matrix_transpose", [](const V &v) { return V{v[0].matrix_transpose()}; }, {x});
    gradcheck("expand_dims", [](const V &v) { return V{v[0].expand_dims(0)}; }, {x});
    gradcheck("squeeze", [](const V &v) { return V{v[0].squeeze(1)}; }, {aux_randn({3, 1, 2})});
    gradcheck("stack", [](const V &v) { return V{Tensor::stack({v[0], v[1]}, 1)}; },
              {aux_randn({3, 2}), aux_randn({3, 2})});
    gradcheck("flip", [](const V &v) { return V{v[0].flip(1)}; }, {x});
    gradcheck("tile", [](const V &v) { return V{v[0].tile({2, 1, 3})}; }, {x});
    gradcheck("diag", [](const V &v) { return V{v[0].diag()}; }, {aux_randn({2, 4, 1})});
    gradcheck("diagonal", [](const V &v) { return V{v[0].diagonal()}; }, {x});
    gradcheck("tril", [](const V &v) { return V{v[0].tril()}; }, {x});
    gradcheck("triu", [](const V &v) { return V{v[0].triu(1)}; }, {x});
    gradcheck("block_diag", [](const V &v) { return V{Tensor::block_diag({v[0], v[1]})}; },
              {aux_randn({2, 3}), aux_randn({1, 2})});
    gradcheck("sort", [](const V &v) { return V{v[0].sort(-1, true)}; }, {x});
    gradcheck("topk", [](const V &v) { return V{v[0].topk(2).first}; }, {x});
}

void linear_algebra() {
    using V = std::vector<Tensor>;
    gradcheck("matmul", [](const V &v) { return V{v[0].matmul(v[1])}; },
              {aux_randn({2, 3, 4}), aux_randn({4, 2})});
    gradcheck("solve", [](const V &v) { return V{aux_regular(v[0]).solve(v[1])}; },
              {aux_randn({4, 4}), aux_randn({4, 2})});
    gradcheck("inv", [](const V &v) { return V{aux_regular(v[0]).inv()}; }, {aux_randn({3, 3})});
    gradcheck("lstsq", [](const V &v) { return V{v[0].lstsq(v[1])}; },
              {aux_randn({6, 3}), aux_randn({6, 2})});
    gradcheck("pinv", [](const V &v) { return V{v[0].pinv()}; }, {aux_randn({4, 3})});
    gradcheck("cholesky", [](const V &v) { return V{aux_spd(v[0]).cholesky()}; },
              {aux_randn({3, 3})});
    gradcheck("qr",
              [](const V &v) {
                  const auto [Q, R] = v[0].qr();
                  return V{Q, R};
              },
              {aux_randn({5, 3})});
    gradcheck("svd",
              [](const V &v) {
                  const auto [U, S, W] = v[0].svd();
                  return V{U, S, W};
              },
              {aux_randn({4, 3})});
    gradcheck("eigh",
              [](const V &v) {
                  const auto [w, Q] = (v[0] + v[0].matrix_transpose()).eigh();
                  return V{w, Q};
              },
              {aux_randn({4, 4})});
    gradcheck("trace", [](const V &v) { return V{v[0].trace()}; }, {aux_randn({2, 3, 3})});
    gradcheck("kron", [](const V &v) { return V{v[0].kron(v[1])}; },
              {aux_randn({2, 3}), aux_randn({2, 2})});
    gradcheck("expm", [](const V &v) { return V{v[0].expm()}; }, {aux_randn({2, 3, 3})});
    gradcheck("matrix_norm",
              [](const V &v) {
                  return V{v[0].matrix_norm(Norm::one), v[0].matrix_norm(Norm::inf),
                           v[0].matrix_norm(Norm::fro), v[0].matrix_norm(Norm::two)};
              },
              {aux_randn({4, 3})});
    gradcheck("vector_norm",
              [](const V &v) {
                  return V{v[0].vector_norm(Norm::one, -1), v[0].vector_norm(Norm::inf, -1),
                           v[0].vector_norm(Norm::two, 0)};
              },
              {aux_away({4, 3})});
}

void fused() {
    using V = std::vector<Tensor>;
    gradcheck("zonotope_linear_map",
              [](const V &v) {
                  const auto [Mc, MG] = Tensor::zonotope_linear_map(v[0], v[1], v[2]);
                  return V{Mc, MG};
              },
              {aux_randn({3, 4}), aux_randn({4, 1}), aux_randn({2, 4, 5})});
    gradcheck("zonotope_interval_map",
              [](const V &v) {
                  const auto [Tc, TG] =
                      Tensor::zonotope_interval_map(v[0], v[0] + v[1].abs(), v[2], v[3]);
                  return V{Tc, TG};
              },
              {aux_randn({3, 3}), aux_away({3, 3}), aux_away({3, 1}), aux_away({3, 4})});
    gradcheck("zonotope_linear_combination",
              [](const V &v) {
                  const auto [c, G] = Tensor::zonotope_linear_combination(v[0], v[1], v[2], v[3]);
                  return V{c, G};
              },
              {aux_randn({3, 1}), aux_randn({3, 4}), aux_randn({3, 1}), aux_randn({3, 4})});
    gradcheck("interval_matmul",
              [](const V &v) {
                  const auto [lo, hi] = Tensor::interval_matmul(v[0], v[0] + v[1].abs(), v[2]);
                  return V{lo, hi};
              },
              {aux_randn({3, 4}), aux_away({3, 4}), aux_away({4, 2})});
    gradcheck("matmul_interval",
              [](const V &v) {
                  const auto [lo, hi] = Tensor::matmul_interval(v[2], v[0], v[0] + v[1].abs());
                  return V{lo, hi};
              },
              {aux_randn({4, 2}), aux_away({4, 2}), aux_away({3, 4})});
    gradcheck("abs_sum", [](const V &v) { return V{v[0].abs_sum()}; }, {aux_away({2, 3, 5})});
}

} // namespace

int main() {
    const BackendScope scope("torch");
    torch::manual_seed(5);
    elementwise();
    reductions();
    shapes();
    linear_algebra();
    fused();
    return test::finish("tensor gradcheck (libtorch)");
}
