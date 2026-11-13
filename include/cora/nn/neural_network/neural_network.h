// neural_network - a feed-forward network of affine layers and activations, on points and sets
//
// Layer k maps h -> a_k(W_k h + b_k) with the activation a_k (identity, ReLU, sigmoid or tanh);
// without activations given, every layer but the last is followed by max(., 0). A set that goes
// through the network comes out as a zonotope (or polynomial zonotope) that contains the outputs
// of all its points: an affine layer maps it exactly, and an activation is enclosed neuron by
// neuron by one line and an error (CORA's poly_method "singh", order 1: the tightest
// parallelogram of a ReLU, [Singh et al., NeurIPS 2018, Thm. 3.1 and 3.2] for sigmoid and tanh).
// On a polynomial zonotope the dependency on its factors stays, evaluate_with_input returns the
// pairs (x, N(x)) of a set, and a neuron may have an order k (NnOptions::orders): its activation
// is enclosed by a polynomial of that order, evaluated on the powers of its set, and refine
// raises the orders where the errors are large. Everything is computed on tensors.
//
// Syntax:     NeuralNetwork nn({{W1, b1}, {W2, b2}});   Zonotope Y = nn.evaluate(X);
//             PolyZonotope XY = nn.evaluate_with_input(PolyZonotope(X));
// Operations: evaluate, evaluate_with_input, approx_errors (evaluate.cpp), refine (refine.cpp),
//             sensitivity (sensitivity.cpp)
// See also:   cora/cont_set/zonotope/zonotope.h, cora/cont_set/poly_zonotope/poly_zonotope.h
//
// CORA: nn/@neuralNetwork/neuralNetwork.m
// CORA: nn/@neuralNetwork/refine.m
// CORA: nn/@neuralNetwork/getRefinableLayers.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Tobias Ladner, Claude Sonnet 5.5
// Written:       02-November-2026
// Last update:   13-November-2026
// Reviewed:      ---

#pragma once

#include "cora/cont_set/poly_zonotope/poly_zonotope.h"
#include "cora/cont_set/zonotope/zonotope.h"

#include <optional>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// The options -------------------------------------------------------------------------------------

/// The function applied to the output of a layer.
enum class Activation { identity, relu, sigmoid, tanh };

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

/// How refine raises the orders (CORA's neuralNetwork/refine).
struct RefineOptions {
    int max_order = 5;
    RefineType type = RefineType::layer;
    RefineHeuristic heuristic = RefineHeuristic::approx_error;
    double gamma = 0.1;
};

// The network -------------------------------------------------------------------------------------

class NeuralNetwork {
  public:
    /// One layer: the weights W (out, in) and the bias b, a column (out, 1).
    struct Layer {
        Tensor W, b;
    };

    /// The network of the layers, first to last; the sizes of consecutive layers must match. Every
    /// layer but the last is followed by a ReLU.
    explicit NeuralNetwork(std::vector<Layer> layers);

    /// The same with the activation of every layer (one per layer, the last one included).
    NeuralNetwork(std::vector<Layer> layers, std::vector<Activation> activations);

    int64_t num_layers() const { return static_cast<int64_t>(layers_.size()); }
    int64_t input_dim() const { return layers_.front().W.shape()[1]; }
    int64_t output_dim() const { return layers_.back().W.shape()[0]; }
    const std::vector<Layer> &layers() const { return layers_; }
    const std::vector<Activation> &activations() const { return activations_; }

    /// The outputs at the points, the columns of x (in, N): (out, N).
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

    /// The Jacobian (out, in) of the network at the point x (in, 1), by the chain rule (CORA's
    /// calcSensitivity of one point, whose mean(abs(S), 1) is the sensitivity of each input).
    Tensor sensitivity(const Tensor &x) const;

  private:
    std::vector<Layer> layers_;
    std::vector<Activation> activations_;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
