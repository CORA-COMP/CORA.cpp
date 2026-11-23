// neural_network - a feed-forward network of affine layers and activations, on points and sets
//
// Layer k maps h -> a_k(W_k h + b_k) with the activation a_k (identity, ReLU, sigmoid, tanh, x^2,
// sign, exp, 1 / max(x, 1), or verify's product of the halves); without activations given, every
// layer but the last is followed by max(., 0). A set that goes through the network comes out as a
// zonotope (or polynomial zonotope) that contains the outputs of all its points: an affine layer
// maps it exactly, and an activation is enclosed neuron by neuron by one line and an error (CORA's
// poly_method "singh", order 1: the tightest parallelogram of a ReLU, [Singh et al., NeurIPS 2018,
// Thm. 3.1 and 3.2] for sigmoid and tanh; for x^2 the secant slope l + u). On a polynomial
// zonotope the factors stay, evaluate_with_input returns the pairs (x, N(x)), and a neuron may
// have an order k (NnOptions::orders), raised by refine. For points and verify, a layer may be a
// convolution or a gather (conv2d.cpp) and form a graph (a ResNet's residuals).
//
// Syntax:     NeuralNetwork nn({{W1, b1}, {W2, b2}});   Zonotope Y = nn.evaluate(X);
//             PolyZonotope XY = nn.evaluate_with_input(PolyZonotope(X));
// Operations: evaluate, evaluate_with_input, approx_errors (evaluate.cpp), refine (refine.cpp),
//             sensitivity, verify, evaluate_zonotope_batch, backprop (one file each), layers
// See also:   cora/cont_set/zonotope/zonotope.h, cora/cont_set/poly_zonotope/poly_zonotope.h
// References:
//    [1] T. Ladner, M. Althoff. "Automatic abstraction refinement in neural network verification
//        using sensitivity analysis", HSCC 2023.
//    [2] L. Koller, T. Ladner, M. Althoff. "Out of the shadows: exploring a latent space for
//        neural network verification", ICLR 2026 (arXiv:2505.17854).
//
// CORA: nn/@neuralNetwork/neuralNetwork.m
// CORA: nn/@neuralNetwork/refine.m
// CORA: nn/@neuralNetwork/getRefinableLayers.m
// CORA: nn/@neuralNetwork/getNumNeurons.m
// CORA: nn/@neuralNetwork/getNumParams.m
// CORA: nn/layers/linear/nnLinearLayer.m
// CORA: nn/layers/linear/nnIdentityLayer.m
// CORA: nn/layers/linear/nnElementwiseAffineLayer.m
// CORA: nn/layers/linear/nnBatchNormLayer.m
// CORA: nn/layers/linear/nnAvgPool2DLayer.m
// CORA: nn/layers/linear/nnLipConstrLinearLayer.m
// CORA: nn/layers/nonlinear/nnExpLayer.m
// CORA: nn/layers/nonlinear/nnSoftmaxLayer.m
// CORA: nn/layers/nonlinear/nnMaxPool2DLayer.m
// CORA: nn/@neuralNetwork/evaluateZonotopeBatch.m
// CORA: nn/@neuralNetwork/backprop.m
// CORA: nn/@neuralNetwork/backpropIntervalBatch.m
// CORA: nn/@neuralNetwork/propagateBounds.m
// CORA: nn/@neuralNetwork/getInputNeuronOrder.m
// CORA: nn/@neuralNetwork/getNormalForm.m
// CORA: nn/@neuralNetwork/verify.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Tobias Ladner, Claude Sonnet 5.5, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   23-November-2026
// Reviewed:      ---

#pragma once

#include "cora/cont_set/poly_zonotope/poly_zonotope.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/nn/neural_network/verify.h"

#include <array>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// The activations ---------------------------------------------------------------------------------

/// A layer's function: square x^2, sign, reciprocal 1 / max(x, 1) (a softmax's sums), product [a o
/// b; 0] of the halves [a; b] (verify only), floor, cube x^3, inverse 1 / x (x > 0; nn4sys' x^3 /
/// sum x^3), sin and cos (ml4acopf), softplus log(1 + e^x), root sqrt x (x >= 0), inv_sqrt_root
/// x^(-1/2) (x > 0), group_sort (points only: Layer::group_size).
enum class Activation {
    identity,
    relu,
    sigmoid,
    tanh,
    square,
    sign,
    exp,
    reciprocal,
    product,
    floor,
    cube,
    inverse,
    sin,
    cos,
    softplus,
    root,
    inv_sqrt_root,
    group_sort
};

/// The polynomial over a neuron's range (CORA's poly_method): at order 1 singh or regression; of
/// higher order (polynomial zonotopes only) any, singh only for a ReLU of order 2.
enum class PolyMethod {
    singh,
    regression,
    ridge_regression,
    bernstein,
    taylor,
    bounds,
    throw_catch
};

/// What CORA's refine raises: all neurons, the layer of the largest heuristic, or the neurons of
/// the layer of the neuron of the largest heuristic whose heuristic is above gamma times it.
enum class RefineType { all, layer, neuron };

/// The heuristic of refine: the approximation errors, those divided by the (1-based) index of the
/// refinable layer, or none (every neuron counts, with type all).
enum class RefineHeuristic { approx_error, layer_bias, all };

// The options -------------------------------------------------------------------------------------

/// The error of the regression line, CORALean's float definitions: extremum (CORA's), that of the
/// ends and the stationary points; bracket, the looser one of the derivative bracket.
enum class RegressionError { extremum, bracket };

/// What evaluate_with_input does with the error of the activations (CORA's options.nn).
struct NnOptions {
    /// true: the errors are independent generators (add_approx_error_to_GI); false: new factors
    /// with their own identifiers, dependent generators.
    bool add_approx_error_to_gi = true;
    PolyMethod poly_method = PolyMethod::singh;
    /// the error of poly_method regression (sigmoid and tanh)
    RegressionError regression_error = RegressionError::extremum;
    /// The order per layer and neuron (CORA's layer.order): none is 1, one value is every
    /// neuron's, a list of the size of the layer is one per neuron.
    std::vector<std::vector<int>> orders = {};
    /// The most generators the sets in the network may have (CORA's num_generators); 0: no limit.
    /// The longest stay, the others become a box added as the error.
    int64_t num_generators = 0;
    /// The most generators after the powers (CORA's max_gens_post); -1: 100 * num_generators.
    int64_t max_gens_post = -1;
    /// Whether the generators are reduced before the powers so that their number stays in reach
    /// of max_gens_post (CORA's do_pre_order_reduction).
    bool do_pre_order_reduction = true;
    /// Whether the independent generators become dependent ones with factors of their own
    /// (CORA's remove_GI); unset: not add_approx_error_to_gi.
    std::optional<bool> remove_gi = std::nullopt;
    /// Whether the errors of the activations are added to the set (CORA's use_approx_error).
    bool use_approx_error = true;
};

// The refinement ----------------------------------------------------------------------------------

/// How refine raises the orders (CORA's neuralNetwork/refine).
struct RefineOptions {
    int max_order = 5;
    RefineType type = RefineType::layer;
    RefineHeuristic heuristic = RefineHeuristic::approx_error;
    double gamma = 0.1;
};

// The batches -------------------------------------------------------------------------------------

/// The options of evaluate_zonotope_batch (CORA's options.nn of evaluateZonotopeBatch).
struct ZonotopeBatchOptions {
    /// The most approximation errors per activation layer that get a generator; the rest go to
    /// the radius dr of the interval center (CORA's num_approx_err; inf: all).
    double num_approx_err = std::numeric_limits<double>::infinity();
    /// The line of an activation: "bounds" (CORA's batch default) or "singh".
    std::string poly_method = "bounds";
    /// Which errors get a generator: "sequential", "length" or "sensitivity*length".
    std::string approx_error_order = "sequential";
};

/// A batch of B zonotopes in rows: the centers c (B, n), the generators G (B, q, n), and the
/// radius dr (B, n) of the interval center (undefined when every error has a generator).
struct ZonotopeBatch {
    Tensor c, G, dr;
};

/// The input boxes of an activation layer's neurons (CORA's layer.l and layer.u): undefined, or
/// with a NaN, for none.
struct NnBounds {
    Tensor l, u;
};

/// How a composite layer joins the outputs of its branches (CORA's aggregation).
enum class Aggregation { add, concat };

/// A polynomial p (highest power first) with |a(x) - p(x)| <= d on [l, u] (CORA's region polys).
struct RegionPoly {
    double l, u;
    std::vector<double> p;
    double d;
};

/// CORA's computeReducedNetwork name-value pairs: dynamic buckets (the shuffled centers) or the
/// static ones, the tolerance (none: 0.01; one, or per layer), input compression, the target rate
/// (none, one, or per layer), the least searches, the seed of the shuffle.
struct ReduceOptions {
    bool dynamic_buckets = false;
    std::vector<double> bucket_tol = {};
    bool input_compression = false;
    std::vector<double> reduction_rate = {};
    int min_searches = 10;
    uint64_t seed = 1;
};

struct ReducedNetwork;

// The network -------------------------------------------------------------------------------------

class NeuralNetwork {
  public:
    /// A 2-D convolution's strides, dilations, pads (top, left, bottom, right) and groups.
    struct Conv2d {
        std::array<int64_t, 2> stride{1, 1}, dilation{1, 1};
        std::array<int64_t, 4> pads{0, 0, 0, 0};
        int64_t groups = 1;
    };

    /// The index lists and kernel matrices of a convolution or a gather (conv2d.cpp).
    struct LinearPlan;

    /// The source of a layer that reads the layer before (Layer::source).
    static constexpr int64_t kPrevious = -2;

    struct Layer;

    /// A residual input of a layer: the output of layer `source` (-1: the network's input)
    /// through its own linear map (null: the identity), added to the layer's linear part.
    struct Residual {
        int64_t source = -1;
        std::shared_ptr<const Layer> map = nullptr;
    };

    /// One layer: the weights W (out, in) and the bias b (out), or a convolution or a gather
    /// (plan, never a matrix: conv2d.cpp), of the output of layer source (-1: the input), plus
    /// the residuals' maps of their sources (the network adds their biases into b).
    struct Layer {
        Tensor W, b;
        std::shared_ptr<const LinearPlan> plan = nullptr;
        int64_t source = kPrevious;
        std::vector<Residual> residuals = {};
        /// The group of Activation::group_sort: consecutive neurons sorted ascending together.
        int64_t group_size = 2;

        /// ONNX's Conv of the input (channels, height, width) in C order: kernel (out, in /
        /// groups, kh, kw), the bias one entry per output (out channels times pixels).
        static Layer conv2d(const Tensor &kernel, const Tensor &bias,
                            const std::vector<int64_t> &input, const Conv2d &spec,
                            bool check_bias = true);
        /// ONNX's ConvTranspose (kernel (in, out / groups, kh, kw)): the transpose of the conv2d
        /// from its output (channels, height, width) back to its input, with the same spec.
        static Layer conv_transpose2d(const Tensor &kernel, const Tensor &bias,
                                      const std::vector<int64_t> &output, const Conv2d &spec);
        /// Output i is input index[i] (negative: 0) times scale[i] (undefined: 1), plus the bias;
        /// an input may repeat.
        static Layer gather(const std::vector<int64_t> &index, int64_t inputs, const Tensor &bias,
                            const Tensor &scale = Tensor());

        /// CORA's nnLinearLayer: W (out, in) and b (out).
        static Layer linear(const Tensor &W, const Tensor &b);
        /// CORA's nnIdentityLayer on n values.
        static Layer identity(int64_t n);
        /// CORA's nnElementwiseAffineLayer: scale .* x + offset on n values; scale and offset are
        /// one entry (all values) or n entries.
        static Layer elementwise_affine(const Tensor &scale, const Tensor &offset, int64_t n);
        /// CORA's nnBatchNormLayer in inference mode on the input (channels, height, width):
        /// (x - mean) / sqrt(var + epsilon) .* scale + offset per channel, folded into one map.
        static Layer batch_norm(const Tensor &scale, const Tensor &offset, const Tensor &mean,
                                const Tensor &var, const std::vector<int64_t> &input,
                                double epsilon = numerics::kBatchNormEpsilon);
        /// CORA's nnAvgPool2DLayer: the mean of each pool window per channel (a depthwise
        /// convolution); an empty stride is the pool size.
        static Layer avg_pool2d(const std::array<int64_t, 2> &pool,
                                const std::vector<int64_t> &input,
                                const std::array<int64_t, 2> &stride = {0, 0},
                                const std::array<int64_t, 4> &pads = {0, 0, 0, 0});
        /// CORA's nnLipConstrLinearLayer: N (W x + b), N = diag(1 / max(1, |W| 1 / lambda)).
        static Layer lip_constr_linear(const Tensor &W, const Tensor &b, double lambda = 1.0);

        int64_t inputs() const;
        int64_t outputs() const;
        /// The entries per row of a product's largest intermediate (a convolution's patches).
        int64_t work() const;
        /// W times each row of x (..., in) without the bias; |W|; W^T times rows (..., out); |W|^T.
        Tensor rows(const Tensor &x) const { return aux_product(x, false, false); }
        Tensor abs_rows(const Tensor &x) const { return aux_product(x, true, false); }
        Tensor rows_t(const Tensor &y) const { return aux_product(y, false, true); }
        Tensor abs_rows_t(const Tensor &y) const { return aux_product(y, true, true); }
        /// The layer in the dtype, on the device (empty: where it is).
        Layer to(DType dtype, const std::string &device = "") const;
        /// The matrix W (out, in), built from the unit vectors for a convolution or a gather.
        Tensor dense() const;

      private:
        Tensor aux_product(const Tensor &x, bool absolute, bool transposed) const;
    };

    /// CORA's nonlinear layers that this port lowers to several layers (as the ONNX front end
    /// does), with their activations: a network of the block, or the layers after layer `previous`
    /// of a larger one (their sources are absolute: -1 is the network's input).
    struct Block {
        std::vector<Layer> layers;
        std::vector<Activation> activations;
    };

    /// CORA's nnExpLayer on n values: exp of each, after the layer `previous`.
    static Block exp_layer(int64_t n, int64_t previous = -1);
    /// CORA's nnSoftmaxLayer on n values, exp(x_i - max x) / sum_j exp(x_j - max x): as
    /// 1 / sum_j exp(x_j - x_i), the differences exp layer then the sums and 1 / max(s, 1), which
    /// is the same function (two layers).
    static Block softmax_layer(int64_t n, int64_t previous = -1);
    /// CORA's nnMaxPool2DLayer on the input (channels, height, width) in C order, for a pool
    /// (kh, kw): max(a, b) = b + relu(a - b) over the window's items in pairs, round by round (an
    /// empty stride is the pool size, as CORA's).
    static Block max_pool2d_layer(const std::array<int64_t, 2> &pool,
                                  const std::vector<int64_t> &input,
                                  const std::array<int64_t, 2> &stride = {0, 0},
                                  int64_t previous = -1);

    /// CORA's nnSoftPlusLayer on n values, log(1 + e^(k x)) / k: k x, softplus, 1 / k (sharpness
    /// k > 0; two layers unless k is 1).
    static Block softplus_layer(int64_t n, double sharpness = 1.0, int64_t previous = -1);
    /// CORA's nnRootLayer, sqrt x (x >= 0), and nnInvSqrtRootLayer, x^(-1/2) (x > 0), on n values.
    static Block root_layer(int64_t n, int64_t previous = -1);
    static Block inv_sqrt_root_layer(int64_t n, int64_t previous = -1);
    /// CORA's nnGroupSortLayer on n values: groups of group_size consecutive neurons sorted
    /// ascending. Points and backprop only (CORA has no set version either).
    static Block group_sort_layer(int64_t n, int64_t group_size = 2, int64_t previous = -1);
    /// CORA's nnCompositeLayer: the branches (blocks whose sources are -1 for the input, or
    /// their own layers) on the same input of input_size values, their outputs added or
    /// concatenated along concat_dim of the output shapes (default: flat vectors), as a graph.
    static Block composite_layer(const std::vector<Block> &branches, Aggregation aggregation,
                                 int64_t input_size,
                                 const std::vector<std::vector<int64_t>> &output_shapes = {},
                                 int64_t concat_dim = 0, int64_t previous = -1);
    /// CORA's nnGeneratorReductionLayer.evaluateZonotopeBatch: of the batch's generators keep the
    /// max_gens - n first of the ascending lengths (CORA's order), the others become the box
    /// diag(sum |g|); (B, max_gens, n). A batch with at most max_gens - n generators is kept.
    static ZonotopeBatch reduce_generators(const ZonotopeBatch &Z, int64_t max_gens);
    /// CORA's getMergeBuckets of an activation layer: its values at -inf and inf that are finite
    /// (the ReLU's: 0).
    static std::vector<double> merge_buckets(Activation activation);
    /// CORA's findRegionPolys: regions of [-inf, inf] with a polynomial of the order each (the
    /// regression fit) whose error is below tol, from (pStart, dStart) below l_max to (pEnd, dEnd)
    /// above u_max.
    static std::vector<RegionPoly>
    find_region_polys(Activation activation, double tol, int order, double l_max, double u_max,
                      const std::vector<double> &p_start, double d_start,
                      const std::vector<double> &p_end, double d_end);

    /// The network of a block.
    explicit NeuralNetwork(const Block &block) : NeuralNetwork(block.layers, block.activations) {}

    /// The network of the layers, first to last; each layer's inputs must match the outputs of
    /// its source and residuals, and every layer but the last must be read by a later one. Every
    /// layer but the last is followed by a ReLU.
    explicit NeuralNetwork(std::vector<Layer> layers);

    /// The same with the activation of every layer (one per layer, the last one included).
    NeuralNetwork(std::vector<Layer> layers, std::vector<Activation> activations);

    int64_t num_layers() const { return static_cast<int64_t>(layers_.size()); }
    int64_t input_dim() const { return layers_.front().inputs(); }
    int64_t output_dim() const { return layers_.back().outputs(); }
    /// Whether every layer is a matrix (the set operations other than verify need that).
    bool is_dense() const;
    /// Whether every layer reads the layer before only (the set operations other than verify and
    /// the points need that).
    bool is_chain() const;
    /// The output layer k reads: the index of a layer, -1 for the network's input.
    int64_t source(std::size_t k) const;
    const std::vector<Layer> &layers() const { return layers_; }
    const std::vector<Activation> &activations() const { return activations_; }

    /// The number of neurons in front of each layer and after the last, (num_layers + 1)
    /// entries (CORA's getNumNeurons).
    std::vector<int64_t> num_neurons() const;

    /// The learnable parameters of a dense network: the entries of all W and b (CORA's
    /// getNumParams).
    int64_t num_params() const;

    /// The outputs at the points, the rows of x (N, in): (N, out).
    Tensor evaluate(const Tensor &x) const;

    /// A zonotope that contains the outputs of every point of X (which may be a batch). Only the
    /// poly_method of the options is used.
    Zonotope evaluate(const Zonotope &X, const NnOptions &options = {}) const;

    /// A polynomial zonotope that contains the outputs of every point of X: the output rows of
    /// evaluate_with_input.
    PolyZonotope evaluate(const PolyZonotope &X, const NnOptions &options = {}) const;

    /// The set of the pairs (x, N(x)) for x in X, a zonotope of dimension in + out: the generators
    /// of X stay the first ones, so the output depends on the same factors (CORA's aux_appendU2X).
    /// The generator of an exact neuron is not added.
    Zonotope evaluate_with_input(const Zonotope &X, const NnOptions &options = {}) const;

    /// The same for a polynomial zonotope: the factors and the exponents of X stay, the outputs
    /// are exact linear images in them, and the errors of the activations are added by options.
    PolyZonotope evaluate_with_input(const PolyZonotope &X, const NnOptions &options = {}) const;

    /// A batch of zonotopes through the network, an activation by its line (CORA's
    /// evaluateZonotopeBatch): the sets (B, n) and (B, q, n), a graph or a chain, any layer.
    ZonotopeBatch evaluate_zonotope_batch(const Tensor &c, const Tensor &G,
                                          const ZonotopeBatchOptions &options = {}) const;

    /// The gradient of the loss at the input (B, in) for the loss's gradient grad_out (B, out) at
    /// the output of the network at the points x (B, in) (CORA's backprop; the weights are not
    /// updated). A chain of any layers but a product.
    Tensor backprop(const Tensor &x, const Tensor &grad_out) const;

    /// The bounds [gl, gu] (B, in) of the gradients at the input over the box [l, u] (B, in)
    /// from the bounds (B, out) at the output (CORA's backpropIntervalBatch, Gowal et al. 2019):
    /// each layer's input box is that of interval arithmetic.
    std::pair<Tensor, Tensor> backprop_interval_batch(const Tensor &l, const Tensor &u,
                                                      const Tensor &gl, const Tensor &gu) const;

    /// CORA's propagateBounds as a function: the bounds (one per layer; those of a layer without
    /// an activation are never read) after the bounds of layer i's activation input went through
    /// the layers up to the next activation, whose bounds are set or intersected.
    std::vector<NnBounds> propagate_bounds(std::vector<NnBounds> bounds, int64_t i) const;

    /// CORA's getInputNeuronOrder (0-based, a pixel is y * W + x): "in-order", "snake" or
    /// "sensitivity" (the least mean absolute Jacobian at x first); input_size is (C, H, W).
    std::vector<int64_t> input_neuron_order(const std::string &method, const Tensor &x,
                                            const std::vector<int64_t> &input_size = {}) const;

    /// The same function with the linear layers merged, so that every activation follows one
    /// linear layer (CORA's getNormalForm); a chain of dense layers or convolutions.
    NeuralNetwork normal_form() const;

    /// CORA's computeReducedNetwork of the normal form: neurons whose output range lies within a
    /// bucket's tolerance are merged away, their range added to the next layer as an interval;
    /// the set X through the reduced network is in the result (a chain of ReLU, sigmoid, tanh).
    ReducedNetwork compute_reduced_network(const Zonotope &X,
                                           const ReduceOptions &options = {}) const;

    /// The layers that refine can raise: those with an activation other than the identity (CORA's
    /// getRefinableLayers), as indices into layers().
    std::vector<int64_t> refinable_layers() const;

    /// The approximation errors of the activations when X goes through the network: per layer,
    /// the radius of each neuron (CORA's refine_heu after evaluate; empty for a layer that is not
    /// refinable).
    std::vector<std::vector<double>> approx_errors(const PolyZonotope &X,
                                                   const NnOptions &options = {}) const;

    /// The orders after one step of refine: orders as NnOptions::orders, heuristic the values per
    /// layer and neuron (approx_errors); the result has one entry per neuron of every layer.
    std::vector<std::vector<int>> refine(const std::vector<std::vector<int>> &orders,
                                         const std::vector<std::vector<double>> &heuristic,
                                         const RefineOptions &options = {}) const;

    /// The Jacobian (out, in) of the network at the point x (in), by the chain rule (CORA's
    /// calcSensitivity of one point, whose mean(abs(S), 1) is the sensitivity of each input).
    Tensor sensitivity(const Tensor &x) const;

    /// CORA's verify of a VNN-COMP property within timeout seconds: "sat" with a counterexample
    /// that confirm accepts (unset: the float64 output check), "unsat" or "unknown" (verify.h).
    NnVerifyResult verify(const NnProperty &property, double timeout,
                          const NnVerifyOptions &options = {}, const ConfirmFn &confirm = {}) const;

  private:
    std::vector<Layer> layers_;
    std::vector<Activation> activations_;
};

/// The result of compute_reduced_network: the merged layers (biases are CORA's b), each layer's
/// error interval d = center +- radius (CORA's layer.d, added after its affine map), the neurons
/// merged per layer (1), the rate of neurons that stay, and the set X through the reduced network.
struct ReducedNetwork {
    NeuralNetwork net;
    std::vector<Tensor> offset_center, offset_radius, merged;
    std::vector<double> reduction_rate;
    Zonotope output;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
