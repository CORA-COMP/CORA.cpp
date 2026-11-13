// neural_network - the oracle's layers and networks on a zonotope (CORALean)
//
// Syntax:     lean::Zonotope Y = lean::evaluate_network({lean::NnLayer::linear(W, b),
//                                                       lean::NnLayer::relu()}, X, "singh");
// Meaning:    nnLayer.evaluateZonotope and neuralNetwork.evaluate: the layers of CORALean in order
//             on the zonotope, each activation by its relaxation at poly_method "singh" (CORA's
//             line) or "curvature" (CORALean's tighter sigmoid and tanh)
// See also:   lean/request.h, cora/nn/neural_network/neural_network.h
//
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/global/oracle/zonotope.h"

#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

/// One layer of the oracle: a kind and its weights.
struct NnLayer {
    std::string kind; // "linear", "relu", "leakyRelu", "sigmoid" or "tanh"
    cora::Tensor W, b;
    double alpha = 0.01; // leakyRelu
    int k = 3;           // sigmoid, tanh: the depth of the exponential's reduction

    static NnLayer linear(const cora::Tensor &W, const cora::Tensor &b);
    static NnLayer relu();
    static NnLayer leaky_relu(double alpha);
    static NnLayer sigmoid();
    static NnLayer tanh();
};

/// The layers as the wire's list, and in `dtype` the scalar type of their weights.
Json layers_json(const std::vector<NnLayer> &layers, std::string &dtype);

/// The layer on the zonotope; regression_error ("bracket" or "extremum", empty: the oracle's
/// default) applies to poly_method "regression" only.
Zonotope evaluate_layer(const NnLayer &layer, const Zonotope &Z,
                        const std::string &poly_method = "singh",
                        const std::string &regression_error = "");

/// The layers in order on the zonotope.
Zonotope evaluate_network(const std::vector<NnLayer> &layers, const Zonotope &Z,
                          const std::string &poly_method = "singh",
                          const std::string &regression_error = "");

/// The exact parts of a polynomial zonotope: the center, the dependent and independent generators.
struct NnPolyResult {
    cora::Tensor c, G, GI;
};

/// The layers in order on the polynomial zonotope (c, G, GI) of the identity exponents, with the
/// error of each activation in the box ("box") or in independent generators ("GI").
NnPolyResult evaluate_network_poly(const std::vector<NnLayer> &layers, const cora::Tensor &c,
                                   const cora::Tensor &G, const cora::Tensor &GI,
                                   const std::string &poly_method, const std::string &error_form,
                                   const std::string &regression_error = "");

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
