// priv_layers - the rows of a set that goes through the layers, and the layer of order k
//
// The state of an evaluation: the rows of the set (centre, dependent generators with the mask of
// those whose exponents are all even, independent generators) that the layers map. A layer whose
// neurons have an order above 1 (or that reduces the generators) is evaluated neuron by neuron on
// the host, which needs the exponents and the identifiers of the factors; the rows then track
// them (track), and an independent generator remembers the column of the input it came from.
//
// Syntax:   priv_layer_poly(rows, activation, orders, errors_in_gi, options, &errors);
// Inputs:   rows - after the affine map of the layer
//           orders - one per neuron
// Outputs:  rows - after the activation
//           errors - the radius of each neuron's error
// See also: evaluate.cpp, priv_poly_power.h, priv_activation_poly.h
//
// CORA: nn/layers/nonlinear/@nnActivationLayer/evaluatePolyZonotope.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Claude Sonnet 5.5
// Written:       13-November-2026
// Last update:   23-November-2026
// Reviewed:      ---

#pragma once

#include "cora/nn/neural_network/neural_network.h"

#include <exception>
#include <utility>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The line of an activation over a range: y in slope x + offset +- radius.
struct PrivLine {
    Tensor slope, offset, radius;
};

/// The rows of a set that goes through the layers: the center, the dependent generators, the mask
/// of those with only even exponents (1, h), the independent generators. Tracked rows also hold
/// the exponents, the factors' identifiers, and the input column of each independent generator.
struct PrivRows {
    Tensor c, G, GI, even;
    int64_t new_factors = 0;
    bool track = false;
    Exponents E;
    std::vector<int64_t> ids, gi_src;
    int64_t last_id = 0;
};

/// The bounds of every row, summed as CORALean's PolyZonotope.interval.
std::pair<Tensor, Tensor> priv_bounds(const PrivRows &r);

/// The values of the activation at the points, the neurons along axis (a product pairs neuron i
/// with neuron i + n / 2 of that axis).
Tensor priv_apply(Activation activation, const Tensor &x, int axis = -1);

/// A product layer's pre-activation [a; b] along the last axis: its halves a and b.
std::pair<Tensor, Tensor> priv_halves(const Tensor &x, int axis = -1);

/// The linear part of a line on rows x (..., n): m x for a neuron by neuron activation; for a
/// product [m_a a + m_b b; 0] with m = [m_a; m_b] (the transpose priv_slope_times_t: [m_a g_a;
/// m_b g_a] of g = [g_a; g_b]).
Tensor priv_slope_times(Activation activation, const Tensor &m, const Tensor &x);
Tensor priv_slope_times_t(Activation activation, const Tensor &m, const Tensor &g);

/// The derivative at the points (..., n) as a slope of priv_slope_times_t: f'(x), and for a
/// product [b; a] (a ReLU's is 0 at 0, as autograd's; sign's 0).
Tensor priv_derivative(Activation activation, const Tensor &x);

/// The line of a product layer on its input zonotope c (B, n), G (B, q, n), dr (B, n) (or none):
/// slopes [c_b; c_a], the rest (G_a beta)(G_b beta) by two scaled squares (docs/nn_verification.md,
/// "Products"), widened by the rounding; the second half's line is 0.
PrivLine priv_product_line(const Tensor &c, const Tensor &G, const Tensor &dr);

/// The line of the activation over the ranges by the method of the options (singh or regression).
PrivLine priv_line(Activation activation, const Tensor &l, const Tensor &u,
                   const NnOptions &options);

/// The line of x^2 over [l, u]: the secant slope l + u, half way between the chord and the parallel
/// tangent, the radius (u - l)^2 / 8 widened by the rounding (sound in floats).
PrivLine priv_square_line(const Tensor &l, const Tensor &u);

/// The line of sign over [l, u]: slope 0, the middle and half the gap of sign(l) and sign(u).
PrivLine priv_sign_line(const Tensor &l, const Tensor &u);

/// The line of exp over [l, u]: the chord's slope m, the middle and half the range of the convex
/// exp(x) - m x (largest at an end, least at log m in [l, u]), widened by the rounding.
PrivLine priv_exp_line(const Tensor &l, const Tensor &u);

/// The line of 1 / max(x, 1) over [l, u]: the chord's slope m on [max(l, 1), max(u, 1)]; f - m x
/// rises to 1, then is convex: largest at l, u or 1, least at l or sqrt(l u) (where f' = m).
PrivLine priv_reciprocal_line(const Tensor &l, const Tensor &u);

/// The message of a bound beyond the float range (an exp that overflows): no line exists there.
constexpr const char *kOutOfRangeMessage = "verify: a bound left the float range";

/// Fails with kOutOfRangeMessage unless every entry of l and u is finite.
void priv_check_finite(const Tensor &l, const Tensor &u);

/// Whether the error is priv_check_finite's.
bool priv_is_out_of_range(const std::exception &error);

/// The mask (1, h) of the generators whose exponents are all even.
Tensor priv_even_mask(const Tensor &like, const Exponents &E);

/// The layer of order k on tracked rows: reduction, the polynomial of every neuron, the errors.
void priv_layer_poly(PrivRows &r, Activation activation, const std::vector<int> &orders,
                     bool errors_in_gi, const NnOptions &options, std::vector<double> *errors);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
