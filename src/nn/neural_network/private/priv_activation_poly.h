// priv_activation_poly - the polynomial that encloses an activation over a range, and its error
//
// CORA's computeApproxPoly of an activation layer for the order k of a neuron: the polynomial p of
// the poly_method over [l, u] (coefficients from the highest power down) and the radius d with
// |a(x) - p(x)| <= d on the range. The error is CORA's computeApproxError: the ReLU's is exact (the
// extrema of p - a on the two linear pieces), that of a sigmoid or tanh the maximum over a grid
// widened by a Lipschitz bound (minMaxDiffOrder), in place of CORA's tables of region polynomials.
//
// Syntax:   PrivApprox a = priv_approx_poly(Activation::sigmoid, l, u, 3, PolyMethod::regression);
// Inputs:   activation, l <= u, order >= 1, method
// Outputs:  coeffs (highest power first, order + 1 entries; fewer where the range is linear), d
// See also: evaluate.cpp, neural_network.h
//
// CORA: nn/layers/nonlinear/@nnActivationLayer/computeApproxPoly.m
// CORA: nn/layers/nonlinear/@nnActivationLayer/nnActivationLayer.m
// CORA authors: Tobias Ladner, Niklas Kochdumper, Sebastian Sigl
// Authors:       Claude Sonnet 5.5
// Written:       13-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/nn/neural_network/neural_network.h"

#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// A polynomial (highest power first) and the radius of its error over a range.
struct PrivApprox {
    std::vector<double> coeffs;
    double d = 0.0;
};

/// The polynomial of the method and order over [l, u] for the activation, and its error radius.
PrivApprox priv_approx_poly(Activation activation, double l, double u, int order,
                            PolyMethod method);

/// The values of p (highest power first) at x.
double priv_polyval(const std::vector<double> &p, double x);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
