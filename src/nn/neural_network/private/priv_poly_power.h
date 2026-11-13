// priv_poly_power - a polynomial of one neuron evaluated on its polynomial zonotope
//
// CORA's evaluatePolyZonotopeNeuron after the fit: p(x) = sum_i a_i x^i for x in c + G + GI, where
// the power x^i is the product x^{i1} x^{i2} (i1 = floor(i / 2), i2 = ceil(i / 2)) of the powers
// below it, each kept with all the generators of the powers it is made of. The product of two
// such sets is calcSquared (dependent generators: all products of two, centre and independent
// generators: the squares of independent generators shifted by half), and the weighted powers are
// added back into the generators of the lower powers, so the result has the layout
// [G (h), G^2, ..., G^k] and [GI (q), GI^2, ..., GI^k]. The cross term of a dependent generator
// with an independent one is the full product, where CORA halves it for exponents that are all
// even (an unsound step: x = a^2 + b squared reaches 4, CORA's set only 3).
//
// Syntax:   PrivNeuron y = priv_eval_poly(c, G, GI, coeffs);   E2 = priv_power_exponents(E, k);
// Inputs:   c, G (h), GI (q) - the neuron's row
//           coeffs - highest power first
//           E, k - the exponents and the order
// Outputs:  the centre, the generators (priv_power_dep_size, priv_power_ind_size), the exponents
// See also: evaluate.cpp, priv_activation_poly.h
//
// CORA: nn/layers/nonlinear/@nnActivationLayer/evaluatePolyZonotopeNeuron.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       13-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/poly_zonotope/poly_zonotope.h"

#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The centre and the generators of a neuron after its polynomial.
struct PrivNeuron {
    double c = 0.0;
    std::vector<double> G, GI;
};

/// p(x) for x in the set c + G + GI of one row, p from the highest power down (order = size - 1).
PrivNeuron priv_eval_poly(double c, const std::vector<double> &G, const std::vector<double> &GI,
                          const std::vector<double> &coeffs);

/// The exponents of the dependent generators after the powers up to order: [E, E^2, ..., E^order].
Exponents priv_power_exponents(const Exponents &E, int order);

/// The number of dependent generators after the powers of h up to order.
int64_t priv_power_dep_size(int64_t h, int order);

/// The number of independent generators after the powers of (h dependent, q independent).
int64_t priv_power_ind_size(int64_t h, int64_t q, int order);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
