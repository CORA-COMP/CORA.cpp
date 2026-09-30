// neuralNetwork - a feed-forward network of affine layers and ReLUs, evaluated on points and sets
//
// Layer k maps h -> W_k h + b_k; every layer but the last is followed by max(., 0). A set that
// goes through the network comes out as a zonotope that contains the outputs of all its points:
// an affine layer maps a zonotope exactly, and a ReLU is enclosed neuron by neuron by the
// tightest parallelogram of one slope (the "zonotope" method of DeepZ). Everything is computed
// on tensors, so gradients flow to the weights and to the input set on libtorch.
//
// Syntax:     NeuralNetwork nn({{W1, b1}, {W2, b2}});   Zonotope Y = nn.evaluate(X);
// Operations: evaluate (one file)
// See also:   contSet/zonotope/zonotope.h

#pragma once

#include "contSet/zonotope/zonotope.h"

#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

class NeuralNetwork {
  public:
    /// One layer: the weights W (out, in) and the bias b, a column (out, 1).
    struct Layer {
        Tensor W, b;
    };

    /// The network of the layers, first to last; the sizes of consecutive layers must match.
    explicit NeuralNetwork(std::vector<Layer> layers);

    int64_t numLayers() const { return static_cast<int64_t>(layers_.size()); }
    int64_t inputDim() const { return layers_.front().W.shape()[1]; }
    int64_t outputDim() const { return layers_.back().W.shape()[0]; }
    const std::vector<Layer> &layers() const { return layers_; }

    /// The outputs at the points, the columns of x (in, N): (out, N).
    Tensor evaluate(const Tensor &x) const;

    /// A zonotope that contains the outputs of every point of X (which may be a batch).
    Zonotope evaluate(const Zonotope &X) const;

  private:
    std::vector<Layer> layers_;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
