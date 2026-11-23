// neural_network - the oracle's layers and networks on a zonotope (CORALean)
//
// Syntax:     lean::Zonotope Y = lean::evaluate_network({lean::NnLayer::linear(W, b),
//                                                       lean::NnLayer::relu()}, X, "singh");
// Meaning:    nnLayer.evaluateZonotope and neuralNetwork.evaluate: the layers of CORALean in order
//             on the zonotope, each activation by its relaxation at poly_method "singh" (CORA's
//             line) or "curvature" (CORALean's tighter sigmoid and tanh); the layer "poly" (the
//             fit of order k given) on a polynomial zonotope by nnLayer.evaluatePolyZonotope;
//             nnLayer.relaxation, the line of an activation over boxes
// See also:   lean/request.h, cora/nn/neural_network/neural_network.h
//
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/global/oracle/poly_zonotope.h"
#include "cora/global/oracle/zonotope.h"
#include "cora/nn/neural_network/neural_network.h"

#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

/// One layer of the oracle: a kind and its weights.
struct NnLayer {
    std::string kind;  // "linear", "relu", "leakyRelu", "sigmoid", "tanh", "sign", "square", "poly"
    cora::Tensor W, b; // linear; poly: W the coefficients (n, k + 1, lowest first), b the errors
    double alpha = 0.01; // leakyRelu
    int k = 3;           // sigmoid, tanh: the depth of the exponential's reduction
    Json geometry;       // conv: the geometry (kshape, input, stride, dilation, pads, groups)

    static NnLayer linear(const cora::Tensor &W, const cora::Tensor &b);
    static NnLayer relu();
    static NnLayer leaky_relu(double alpha);
    static NnLayer sigmoid();
    static NnLayer tanh();
    static NnLayer sign();
    static NnLayer square();
    /// ONNX's Conv of the input (C, H, W): the kernel (out, C / groups, kh, kw), one bias per
    /// output (CORA.cpp's NeuralNetwork::Layer::conv2d).
    static NnLayer conv(const cora::Tensor &kernel, const cora::Tensor &bias,
                        const std::vector<int64_t> &input, const NeuralNetwork::Conv2d &spec);
    static NnLayer poly(const cora::Tensor &coeffs, const cora::Tensor &d);
};

/// A line per neuron: slope x + offset +- error.
struct NnLine {
    Tensor slope, offset, error;
};

/// nnLayer.relaxation: the line of `kind` ("sign", "square", "exp", "reciprocal" or "relu") over
/// each box [l_i, u_i] (columns (n, 1)); for "relu", alpha (n, 1) the slopes of the neurons that
/// cross zero.
NnLine relaxation(const std::string &kind, const cora::Tensor &l, const cora::Tensor &u,
                  const cora::Tensor &alpha = cora::Tensor());

/// nnLayer.productLine: the line of a_i b_i on the zonotope of center c (2 n, 1), generators G
/// (2 n, q) (a its first n rows), interval radii dr (2 n, 1; undefined: none): slope (2 n), offset,
/// error (n).
NnLine product_line(const cora::Tensor &c, const cora::Tensor &G,
                    const cora::Tensor &dr = cora::Tensor());

/// nnLayer.softmaxSumBounds: the box (inf, sup) of W exp(d) + b for l <= d <= u (columns).
std::pair<Tensor, Tensor> softmax_sum_bounds(const cora::Tensor &W, const cora::Tensor &b,
                                             const cora::Tensor &l, const cora::Tensor &u);

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

/// The layer "poly" on the polynomial zonotope: each neuron's fit by the powers of its row, the
/// equal exponents added, the errors as independent generators.
PolyZonotope evaluate_layer(const NnLayer &layer, const PolyZonotope &P);

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
