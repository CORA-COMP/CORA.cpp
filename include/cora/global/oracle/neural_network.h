// neural_network - the oracle's layers and networks on a zonotope (CORALean)
//
// Syntax:     lean::Zonotope Y = lean::evaluate_network({lean::NnLayer::linear(W, b),
//                                                       lean::NnLayer::relu()}, X, "singh");
//             lean::Zonotope Y = lean::evaluate_graph({{W1, b1}, {W2, b2, -2, {{-1}}}}, X);
// Meaning:    nnLayer.evaluateZonotope and neuralNetwork.evaluate: the layers of CORALean in order
//             on the zonotope, each activation by its relaxation at poly_method "singh" (CORA's
//             line) or "curvature" (CORALean's tighter sigmoid and tanh); the layer "poly" (the
//             fit of order k given) on a polynomial zonotope by nnLayer.evaluatePolyZonotope;
//             nnLayer.relaxation, the line of an activation over boxes; nn_evaluate_taylm the
//             layers on Taylor models (neuralNetwork.evaluateTaylm); neuralNetwork.evaluateGraph:
//             a graph network (sources, residuals) in the order of CORA.cpp's ZonoNet::forward,
//             the radii of the interval center as the error box
// See also:   lean/request.h, cora/nn/neural_network/neural_network.h
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       12-November-2026
// Last update:   24-November-2026
// Reviewed:      ---

#pragma once

#include "cora/global/oracle/poly_zonotope.h"
#include "cora/global/oracle/taylm.h"
#include "cora/global/oracle/zonotope.h"
#include "cora/nn/neural_network/neural_network.h"

#include <array>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

// Layers and their lines --------------------------------------------------------------------------

/// One layer of the oracle: a kind and its weights.
struct NnLayer {
    std::string kind; // "linear", "relu", "leakyRelu", "sigmoid", "tanh", "sign", "square", "poly",
                      // "conv", or a line's (NnLayer::line)
    cora::Tensor W, b;   // linear; poly: W the coefficients (n, k + 1, lowest first), b the errors
    double alpha = 0.01; // leakyRelu
    int k = 3;           // sigmoid, tanh: the depth of the exponential's reduction
    Json geometry;       // conv: the geometry (kshape, input, stride, dilation, pads, groups)
    std::vector<int> order; // evaluate_taylm's activations: none, one, or one per neuron

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
    /// CORA.cpp's activation with a line of its own, the relaxation of nnLayer.relaxation at every
    /// neuron's box: "exp", "reciprocal", "cube", "inverse", "softplus", "root" or "invSqrt"; the
    /// monotone lines take the regression slope at poly_method "regression", else Singh's.
    static NnLayer line(const std::string &kind);
    /// CORA.cpp's port's linear layers as it folds them (fields in `geometry`): scale .* x +
    /// offset; batch norm per channel of (C, H, W); the pool windows' means; the rows of W x + b
    /// over max(|W_i| 1 / lambda, 1).
    static NnLayer elementwise_affine(const cora::Tensor &scale, const cora::Tensor &offset);
    static NnLayer batch_norm(const cora::Tensor &scale, const cora::Tensor &offset,
                              const cora::Tensor &mean, const cora::Tensor &var,
                              const std::vector<int64_t> &input, double epsilon);
    static NnLayer avg_pool(const std::array<int64_t, 2> &pool, const std::vector<int64_t> &input,
                            const std::array<int64_t, 2> &stride,
                            const std::array<int64_t, 4> &pads);
    static NnLayer lip_linear(const cora::Tensor &W, const cora::Tensor &b, double lambda);
};

/// A line per neuron: slope x + offset +- error.
struct NnLine {
    Tensor slope, offset, error;
};

/// nnLayer.relaxation: the line of `kind` ("sign", "square", "relu", "sin", "cos", a kind of
/// NnLayer::line) over each box [l_i, u_i] (columns (n, 1)); relu: alpha (n, 1) the crossing
/// neurons' slopes; softplus, root, invSqrt: method "regression" (empty: the oracle's default) or
/// "singh".
NnLine relaxation(const std::string &kind, const cora::Tensor &l, const cora::Tensor &u,
                  const cora::Tensor &alpha = cora::Tensor(), const std::string &method = "");

/// nnLayer.productLine: the line of a_i b_i on the zonotope of center c (2 n, 1), generators G
/// (2 n, q) (a its first n rows), interval radii dr (2 n, 1; undefined: none): slope (2 n), offset,
/// error (n).
NnLine product_line(const cora::Tensor &c, const cora::Tensor &G,
                    const cora::Tensor &dr = cora::Tensor());

/// nnLayer.softmaxSumBounds: the box (inf, sup) of W exp(d) + b for l <= d <= u (columns).
std::pair<Tensor, Tensor> softmax_sum_bounds(const cora::Tensor &W, const cora::Tensor &b,
                                             const cora::Tensor &l, const cora::Tensor &u);

/// A layer of neuralNetwork.backwardBound: an affine map, or a ReLU with the box [l, u] of its
/// inputs and optionally the slopes alpha of its crossing neurons.
struct CrownLayer {
    std::string kind; // "linear" or "relu"
    cora::Tensor W, b, l, u, alpha;
    static CrownLayer linear(const cora::Tensor &W, const cora::Tensor &b);
    static CrownLayer relu(const cora::Tensor &l, const cora::Tensor &u,
                           const cora::Tensor &alpha = cora::Tensor());
};
/// neuralNetwork.backwardBound: CROWN's lower bound (q, 1) of every row of C (q, m) on the outputs
/// over the box |x' - x| <= r (columns), CORA.cpp's priv_backward_lower.
Tensor backward_bound(const std::vector<CrownLayer> &layers, const cora::Tensor &x,
                      const cora::Tensor &r, const cora::Tensor &C);

// Networks on sets and Taylor models --------------------------------------------------------------

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

/// neuralNetwork.evaluateTaylm: the layers in order on one Taylor model per input (all over the
/// same variables), each activation per neuron by its fit of the layer's order; one model per
/// output. Order 1 of every activation at singh or regression, order 2 of relu at singh.
std::vector<Taylm> nn_evaluate_taylm(const std::vector<NnLayer> &layers,
                                     const std::vector<Taylm> &x,
                                     const std::string &poly_method = "singh",
                                     const std::string &regression_error = "");

/// A residual of a graph layer: the output of layer `source` (-1: the input) through R (undefined:
/// the identity).
struct NnGraphResidual {
    int64_t source = -1;
    cora::Tensor R = {};
};

/// A graph layer: W times the output `source` (-1: the input, kPrevious: the one before), plus b
/// and the residuals, then the activation; its errors as generators ("all") or in the error box.
/// activation: "identity", "relu", "sign", "square" or "product" (of the halves).
struct NnGraphLayer {
    static constexpr int64_t kPrevious = -2;
    cora::Tensor W, b;
    int64_t source = kPrevious;
    std::vector<NnGraphResidual> residuals = {};
    std::string activation = "identity";
    std::string slots = "all";
};

/// The graph network on the zonotope (neuralNetwork.evaluateGraph): the last layer's output, the
/// radii of its interval center in the error box.
Zonotope evaluate_graph(const std::vector<NnGraphLayer> &layers, const Zonotope &Z);

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
