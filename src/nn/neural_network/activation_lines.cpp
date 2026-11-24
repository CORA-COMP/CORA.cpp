// activation_lines - the lines of softplus, root and inverse square root over a neuron's range
//
// The three functions are monotone with a monotone derivative, so a line y in slope x + offset
// +- radius needs no search: f(x) - slope x is convex (softplus, inv_sqrt_root) or concave (root)
// at every slope, its extremes are at the ends and at the one point where f' = slope. That inner
// extremum is taken in closed form, not as f at a rounded stationary point: -(m log m) - t log t
// (t = 1 - m) for softplus, 1 / (4 m) for root, (3/2) c with c = (-2 m)^(1/3) for inv_sqrt_root.
// The slope is Singh's (the smaller derivative at an end) or the regression slope; both take the
// same band. softplus log(1 + e^x) is convex and rising (f' = sigmoid), root sqrt(x) concave and
// rising on x >= 0, inv_sqrt_root sqrt(1 / x) convex and falling on x > 0. CORA's default
// (regression) takes the same slope and a sampled error for softplus; its Singh line of root has
// a wrong derivative (nnRootLayer.getDf: 1/2 f(x), not 1 / (2 f(x))), here the right one.
//
// Syntax:   activation_lines() - the lines, e.g.
//           PrivLine line = priv_monotone_line(Activation::root, l, u, regression);
// Inputs:   activation - softplus, root or inv_sqrt_root
//           l, u - the range of each neuron, l <= u
//           regression - the slope of the least-squares fit (CORA's default), else Singh's
// Outputs:  line - slope, offset and radius per neuron
// See also: cora/nn/neural_network/neural_network.h, evaluate.cpp
// References:
//    [1] G. Singh, T. Gehr, M. Mirman, M. Puschel, M. Vechev. "Fast and effective robustness
//        certification", NeurIPS 2018.
//
// CORA: nn/layers/nonlinear/nnSoftPlusLayer.m
// CORA: nn/layers/nonlinear/nnRootLayer.m
// CORA: nn/layers/nonlinear/nnInvSqrtRootLayer.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       23-November-2026
// Last update:   24-November-2026
// Reviewed:      ---

#include "cora/global/error.h"
#include "nn/neural_network/private/priv_layers.h"

#include <functional>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The functions -----------------------------------------------------------------------------------

/// The function and its derivative.
struct Monotone {
    std::function<Tensor(const Tensor &)> f, derivative;
};

/// softplus log(1 + e^x): f' = 1 / (e^(-x) + 1).
Monotone aux_softplus() {
    return {[](const Tensor &x) { return (x.ones_like() + x.exp()).log(); },
            [](const Tensor &x) { return x.ones_like().divide(x.negative().exp() + 1.0); }};
}

/// sqrt x: f' = 1 / (2 sqrt x).
Monotone aux_root() {
    return {[](const Tensor &x) { return x.sqrt(); },
            [](const Tensor &x) {
                // a point range at 0 has no tangent: the slope 0
                const Tensor zero = x.zeros_like();
                return Tensor::where(x.greater(zero), x.ones_like().divide(x.sqrt()) * 0.5, zero);
            }};
}

/// sqrt(1 / x): f' = -1 / (2 x^(3/2)).
Monotone aux_inv_sqrt() {
    return {[](const Tensor &x) { return x.ones_like().divide(x).sqrt(); },
            [](const Tensor &x) {
                return x.ones_like().divide(x.square().multiply(x).sqrt()).negative() * 0.5;
            }};
}

Monotone aux_function(Activation a) {
    if (a == Activation::softplus) return aux_softplus();
    if (a == Activation::root) return aux_root();
    return aux_inv_sqrt();
}

// The inner extremum ------------------------------------------------------------------------------

/// The band [lo, hi] of f(x) - m x from the ends' band: the closed-form minimum (convex) or maximum
/// (concave) at f'(x) = m folded in where that point exists; elsewhere a harmless slope stands in
/// for m, so that no NaN is computed (CORALean's softplus/root/invSqrtRootLineRelaxation).
std::pair<Tensor, Tensor> aux_band(Activation a, const Tensor &m, const Tensor &lo,
                                   const Tensor &hi) {
    const Tensor zero = m.zeros_like(), one = m.ones_like();
    if (a == Activation::softplus) {
        // 0 < m < 1: at x = log(m / (1 - m)) the minimum -(m log m) - t log t, t = 1 - m
        const Tensor exists = m.greater(zero).multiply(m.less(one));
        const Tensor s = Tensor::where(exists, m, one * 0.5), t = one - s;
        const Tensor inner = s.multiply(s.log()).negative() - t.multiply(t.log());
        return {Tensor::where(exists, lo.minimum(inner), lo), hi};
    }
    if (a == Activation::root) {
        // m > 0: at x = 1 / (4 m^2) the maximum 1 / (4 m)
        const Tensor exists = m.greater(zero);
        const Tensor inner = one.divide(Tensor::where(exists, m, one)) * 0.5 * 0.5;
        return {lo, Tensor::where(exists, hi.maximum(inner), hi)};
    }
    // m < 0: at x = (-2 m)^(-2/3) the minimum c + c / 2, c = (-2 m)^(1/3) = exp(log(-m - m) / 3)
    const Tensor exists = m.less(zero);
    const Tensor s = Tensor::where(exists, m, one.negative()).negative();
    const Tensor c = ((s + s).log() / 3.0).exp();
    return {Tensor::where(exists, lo.minimum(c + c * 0.5), lo), hi};
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The slope and the line --------------------------------------------------------------------------

Tensor priv_regression_slope(const Tensor &l, const Tensor &u,
                             const std::function<Tensor(const Tensor &)> &middle) {
    detail::Entry entry;
    constexpr int64_t kPoints = 20;
    const Tensor zero = l.zeros_like(), one = l.ones_like();
    std::vector<double> index(kPoints), weight(kPoints);
    for (int64_t i = 0; i < kPoints; ++i) {
        index[std::size_t(i)] = double(i);
        weight[std::size_t(i)] = double(2 * i - (kPoints - 1));
    }
    // the points x_i = l + i h, the middles y_i, the weights 2 i - 19; their sum from the right
    const Tensor h = (u - l).divide(one * double(kPoints - 1));
    const Tensor X = l + h.multiply(Tensor::like(l, index, {1, kPoints}));
    const Tensor T = middle(X).multiply(Tensor::like(l, weight, {1, kPoints}));
    Tensor S = zero;
    for (int64_t i = kPoints - 1; i >= 0; --i) S = T.slice(-1, i, i + 1) + S;
    const Tensor empty = u.less_equal(l);
    const Tensor denominator = Tensor::where(empty, one, (u - l) * 70.0);
    return Tensor::where(empty, zero, S.divide(denominator));
}

PrivLine priv_monotone_line(Activation activation, const Tensor &l, const Tensor &u,
                            bool regression) {
    detail::Entry entry;
    CORA_CHECK(activation == Activation::softplus || activation == Activation::root ||
                   activation == Activation::inv_sqrt_root,
               Error::wrong_value,
               [] { return std::string("priv_monotone_line: softplus, root or inv_sqrt_root"); });
    const Tensor zero = l.zeros_like();
    // the domains: root takes x >= 0, inv_sqrt_root x > 0
    const Tensor inside = activation == Activation::root ? l.greater_equal(zero) : l.greater(zero);
    if (activation != Activation::softplus && inside.numel() > 0 &&
        inside.reshape({inside.numel()}).min(0).item() < 1)
        CORA_FAIL(Error::out_of_domain,
                  activation == Activation::root
                      ? "evaluate: root needs x >= 0 at every neuron"
                      : "evaluate: inv_sqrt_root needs x > 0 at every neuron");
    const Monotone fn = aux_function(activation);
    const Tensor fl = fn.f(l), fu = fn.f(u);
    priv_check_finite(fl, fu);
    // the slope: Singh's (the smaller derivative at an end) or the least-squares fit's
    const Tensor m =
        !regression ? (activation == Activation::softplus ? fn.derivative(l) : fn.derivative(u))
                    : priv_regression_slope(l, u, fn.f);
    // f(x) - m x at the ends, then the inner extremum where f' = m inside the line
    const Tensor gl = fl - m.multiply(l), gu = fu - m.multiply(u);
    const auto [lo, hi] = aux_band(activation, m, gl.minimum(gu), gl.maximum(gu));
    return {m, (hi + lo) * 0.5, (hi - lo) * 0.5};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
