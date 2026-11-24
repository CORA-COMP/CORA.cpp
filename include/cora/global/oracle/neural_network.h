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
//             the radii of the interval center as the error box; the forms (merge buckets, group
//             sort, generator reduction, normal form) and the passes of a ReLU chain (propagate
//             bounds, the gradients at a box and at a point, the sensitivity); a zonotope batch's
//             gradients with the lines held, FGSM, the input order and the structure's counts
// See also:   lean/request.h, cora/nn/neural_network/neural_network.h
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       12-November-2026
// Last update:   24-November-2026
// Reviewed:      ---

#pragma once

#include "cora/global/oracle/interval.h"
#include "cora/global/oracle/poly_zonotope.h"
#include "cora/global/oracle/taylm.h"
#include "cora/global/oracle/zonotope.h"
#include "cora/nn/neural_network/neural_network.h"

#include <array>
#include <string>
#include <utility>
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

/// nnLayer.relaxation: the line of `kind` ("sign", "square", "relu", "sin", "cos", an NnLayer::line
/// kind) over each box [l_i, u_i] (columns); relu: alpha the crossing neurons' slopes; softplus,
/// root, invSqrt: method "regression" or "singh" (empty: the oracle's default).
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

// The forms and passes ----------------------------------------------------------------------------

/// nnLayer.mergeBuckets: the finite limits of the activation "relu", "sigmoid" or "tanh".
Tensor merge_buckets(const std::string &kind);

/// nnLayer.groupSort: the point x (n) with each group of `group` consecutive entries ascending.
Tensor group_sort(const cora::Tensor &x, int64_t group);

/// nnLayer.generatorReduction: the max_gens - n shortest generators by |g|_1 kept (ties by index),
/// the others summed into a box.
Zonotope generator_reduction(const Zonotope &Z, int64_t max_gens);

/// A layer of the oracle's normal form: W and b as boxes, and the name of its activation.
struct NnDense {
    Interval W, b;
    std::string activation;
};

/// neuralNetwork.normalForm of a chain of dense layers: the linear layers up to each activation
/// fused, W_k W and W_k b + b_k (activations "relu", "sigmoid", "tanh", "identity" at the end).
std::vector<NnDense> normal_form(const NeuralNetwork &nn);

/// The passes below take a chain of dense layers with ReLU or identity activations. The bounds of
/// propagateBounds are those of each layer's outputs ((n), undefined: none), as NeuralNetwork's.
std::vector<NnBounds> propagate_bounds(const NeuralNetwork &nn, const std::vector<NnBounds> &bounds,
                                       int64_t i);

/// neuralNetwork.backpropIntervalBatch: the boxes of the input gradient's lower and upper bound
/// over the box [l, u] (in) from the bounds [gl, gu] (out) at the output.
std::pair<Interval, Interval> backprop_interval(const NeuralNetwork &nn, const cora::Tensor &l,
                                                const cora::Tensor &u, const cora::Tensor &gl,
                                                const cora::Tensor &gu);

/// neuralNetwork.backprop: the input gradient at the point x (in) of the output gradient g (out).
Interval backprop(const NeuralNetwork &nn, const cora::Tensor &x, const cora::Tensor &g);

/// neuralNetwork.sensitivity: the box of the Jacobian (out, in) at the point x (in).
Interval sensitivity(const NeuralNetwork &nn, const cora::Tensor &x);

// The zonotope batch's gradients and the structure ------------------------------------------------

/// The gradients at an input zonotope's center (n) and generators (n, h), and the slopes of each
/// layer's ReLU lines (undefined for an identity layer).
struct NnZonoGrad {
    Interval g_c, g_gen;
    std::vector<cora::Tensor> slopes;
};

/// neuralNetwork.backpropZonotopeBatch of a ReLU chain with the lines held: the zonotope (c (n),
/// G (n, h)) forward, every error a generator, then grad_c (k) and grad_gen (k, q) backward.
NnZonoGrad backprop_zonotope_batch(const NeuralNetwork &nn, const cora::Tensor &c,
                                   const cora::Tensor &G, const cora::Tensor &grad_c,
                                   const cora::Tensor &grad_gen);

/// nnLayer.backpropZonotopeBatch: a linear layer's W' grad_c and W' grad_gen (c, G unused), or a
/// ReLU's slopes at the bounds of its input set (c, G) times grad_c and G's columns of grad_gen.
NnZonoGrad backprop_zonotope_batch_layer(const NnLayer &layer, const cora::Tensor &c,
                                         const cora::Tensor &G, const cora::Tensor &grad_c,
                                         const cora::Tensor &grad_gen);

/// neuralNetwork.computeFGSMAttack of a ReLU chain: x (n) moved by epsilon along the sign of the
/// gradient of the loss whose derivative at the output is softmax(y) - t.
Interval compute_fgsm_attack(const NeuralNetwork &nn, const cora::Tensor &x, const cora::Tensor &t,
                             double epsilon);

/// neuralNetwork.getInputNeuronOrder (0-based): "in-order", "snake" or "sensitivity" at x; the
/// input size is (C, H, W) or empty.
std::vector<int64_t> input_neuron_order(const NeuralNetwork &nn, const std::string &method,
                                        const cora::Tensor &x,
                                        const std::vector<int64_t> &input_size = {});

/// neuralNetwork.getNumNeurons, getNumParams and getRefinableLayers of a dense chain.
std::vector<int64_t> num_neurons(const NeuralNetwork &nn);
int64_t num_params(const NeuralNetwork &nn);
std::vector<int64_t> refinable_layers(const NeuralNetwork &nn);

/// neuralNetwork.getOrderPattern: the orders of the refinable layers' neurons from those given per
/// layer (none: all 1).
std::vector<std::vector<int64_t>> order_pattern(const NeuralNetwork &nn,
                                                const std::vector<std::vector<int64_t>> &orders);

/// The generator count of a zonotope batch, and per layer the first error generator and the count
/// of them (both 0 for a layer without an activation).
struct ZonoBatchCount {
    int64_t num_gen = 0;
    std::vector<std::pair<int64_t, int64_t>> errors;
};

/// neuralNetwork.prepareForZonoBatchEval: the generators of the input (inf: its width) and of each
/// activation layer's errors (at most num_approx_err; inf: all).
ZonoBatchCount prepare_zono_batch_eval(const NeuralNetwork &nn, double num_init_gens,
                                       double num_approx_err);

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
