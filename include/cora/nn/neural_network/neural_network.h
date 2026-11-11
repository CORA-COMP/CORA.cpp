// neural_network - a feed-forward network of affine layers and activations, on points and sets
//
// Layer k maps h -> a_k(W_k h + b_k) with the activation a_k (identity, ReLU, sigmoid or tanh);
// without activations given, every layer but the last is followed by max(., 0). A set that goes
// through the network comes out as a zonotope (or polynomial zonotope) that contains the outputs
// of all its points: an affine layer maps it exactly, and an activation is enclosed neuron by
// neuron by one line and an error (CORA's poly_method "singh", order 1: the tightest
// parallelogram of a ReLU, [Singh et al., NeurIPS 2018, Thm. 3.1 and 3.2] for sigmoid and tanh).
// On a polynomial zonotope the dependency on its factors stays, and evaluate_with_input returns
// the set of the pairs (x, N(x)) of a set X, which is what a controlled system needs. Everything
// is computed on tensors, so gradients flow to the weights and to the input set on libtorch.
//
// Syntax:     NeuralNetwork nn({{W1, b1}, {W2, b2}});   Zonotope Y = nn.evaluate(X);
//             PolyZonotope XY = nn.evaluate_with_input(PolyZonotope(X));
// Operations: evaluate, evaluate_with_input (evaluate.cpp), sensitivity (sensitivity.cpp)
// See also:   cora/cont_set/zonotope/zonotope.h, cora/cont_set/poly_zonotope/poly_zonotope.h
//
// CORA: nn/@neuralNetwork/neuralNetwork.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/poly_zonotope/poly_zonotope.h"
#include "cora/cont_set/zonotope/zonotope.h"

#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The function applied to the output of a layer.
enum class Activation { identity, relu, sigmoid, tanh };

/// The line over the range of a neuron (CORA's poly_method, order 1): singh, the tightest
/// parallelogram of Singh et al. (NeurIPS 2018), or regression, the least-squares line through 20
/// points with the error that bounds its difference to the activation over the range.
enum class PolyMethod { singh, regression };

/// What evaluate_with_input does with the error of the activations (CORA's options.nn).
struct NnOptions {
    /// true: the errors are independent generators (add_approx_error_to_GI); false: new factors
    /// with their own identifiers, dependent generators.
    bool add_approx_error_to_gi = true;
    PolyMethod poly_method = PolyMethod::singh;
};

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

    /// The Jacobian (out, in) of the network at the point x (in, 1), by the chain rule (CORA's
    /// calcSensitivity of one point, whose mean(abs(S), 1) is the sensitivity of each input).
    Tensor sensitivity(const Tensor &x) const;

  private:
    std::vector<Layer> layers_;
    std::vector<Activation> activations_;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
