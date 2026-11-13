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
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/nn/neural_network/neural_network.h"

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

/// The line of the activation over the ranges by the method of the options (singh or regression).
PrivLine priv_line(Activation activation, const Tensor &l, const Tensor &u,
                   const NnOptions &options);

/// The mask (1, h) of the generators whose exponents are all even.
Tensor priv_even_mask(const Tensor &like, const Exponents &E);

/// The layer of order k on tracked rows: reduction, the polynomial of every neuron, the errors.
void priv_layer_poly(PrivRows &r, Activation activation, const std::vector<int> &orders,
                     bool errors_in_gi, const NnOptions &options, std::vector<double> *errors);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
