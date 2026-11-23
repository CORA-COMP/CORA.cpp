// activation_lines - the lines of softplus, root and inverse square root over a neuron's range
//
// The three functions are monotone with a monotone derivative, so a line y in slope x + offset
// +- radius needs no search: f(x) - slope x is monotone (Singh's parallelogram: the slope is the
// smaller derivative at an end) or convex or concave (the regression slope: its extremes are at
// the ends and at the one point where f' = slope, in closed form). softplus log(1 + e^x) is
// convex and rising (f' = sigmoid), root sqrt(x) concave and rising on x >= 0, inv_sqrt_root
// x^(-1/2) convex and falling on x > 0. CORA's default (regression) takes the same slope and a
// sampled error for softplus, the same closed form for root and inv_sqrt_root; its Singh line of
// root has a wrong derivative (nnRootLayer.getDf: 1/2 f(x), not 1 / (2 f(x))), here the right one.
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
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/error.h"
#include "cora/global/tensor/tolerance.h"
#include "nn/neural_network/private/priv_layers.h"

#include <functional>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The functions -----------------------------------------------------------------------------------

/// The function and the stationary point of f(x) - m x, at the slope m (clamped into the range).
struct Monotone {
    std::function<Tensor(const Tensor &)> f, derivative;
    std::function<Tensor(const Tensor &)> stationary;
    bool convex;
};

/// softplus log(1 + e^x): f' = sigmoid, f'(x) = m at log(m / (1 - m)).
Monotone aux_softplus() {
    return {[](const Tensor &x) { return (x.ones_like() + x.exp()).log(); },
            [](const Tensor &x) { return x.sigmoid(); },
            [](const Tensor &m) {
                const Tensor c = m.maximum(numerics::kTiny).minimum(1.0 - 0x1p-53);
                return c.divide(1.0 - c).log();
            },
            true};
}

/// sqrt x: f' = 1 / (2 sqrt x), f'(x) = m at 1 / (4 m^2) (infinite at m = 0: clamped to u).
Monotone aux_root() {
    return {[](const Tensor &x) { return x.sqrt(); },
            [](const Tensor &x) {
                // a point range at 0 has no tangent: the slope 0
                const Tensor zero = x.zeros_like();
                return Tensor::where(x.greater(zero), x.ones_like().divide(x.sqrt()) * 0.5, zero);
            },
            [](const Tensor &m) { return m.ones_like().divide(m.square() * 4.0); }, false};
}

/// x^(-1/2): f' = -1 / (2 x^(3/2)), f'(x) = m at (-1 / (2 m))^(2/3).
Monotone aux_inv_sqrt() {
    return {[](const Tensor &x) { return x.pow(-0.5); },
            [](const Tensor &x) {
                return x.ones_like().divide(x.square().multiply(x).sqrt()).negative() * 0.5;
            },
            [](const Tensor &m) {
                const Tensor c = m.minimum(-numerics::kTiny);
                return c.ones_like().divide(c * -2.0).pow(2.0 / 3.0);
            },
            true};
}

Monotone aux_function(Activation a) {
    if (a == Activation::softplus) return aux_softplus();
    if (a == Activation::root) return aux_root();
    return aux_inv_sqrt();
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
    if (!regression) {
        // Singh: the smaller derivative at an end; f(x) - slope x is monotone over [l, u]
        const Tensor slope =
            activation == Activation::softplus ? fn.derivative(l) : fn.derivative(u);
        const Tensor offset = (fu + fl - slope.multiply(u + l)) * 0.5;
        // the half range of the monotone f(x) - slope x: the end values in the order of f
        const Tensor rise = (fu - fl - slope.multiply(u - l)) * 0.5;
        return {slope, offset, activation == Activation::inv_sqrt_root ? rise.negative() : rise};
    }
    // the regression slope; f(x) - m x is convex or concave: ends at one side, f' = m at the other
    const Tensor m = priv_regression_slope(l, u, fn.f);
    const auto g = [&](const Tensor &x) { return fn.f(x) - m.multiply(x); };
    const Tensor ends_lo = g(l).minimum(g(u)), ends_hi = g(l).maximum(g(u));
    const Tensor inner = g(fn.stationary(m).maximum(l).minimum(u));
    const Tensor lo = fn.convex ? ends_lo.minimum(inner) : ends_lo;
    const Tensor hi = fn.convex ? ends_hi : ends_hi.maximum(inner);
    return {m, (lo + hi) * 0.5, (hi - lo) * 0.5};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
