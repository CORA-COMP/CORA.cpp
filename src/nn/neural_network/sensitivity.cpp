// sensitivity - the Jacobian of a network at a point
//
// The chain rule through the layers: J_k = diag(a_k'(z_k)) W_k J_{k-1} with z_k = W_k h_{k-1} +
// b_k, from J_0 = I. A ReLU has the derivative 1 where its input is positive and 0 elsewhere
// (CORA's getDf(1) of nnLeakyReLULayer), sigmoid and tanh the derivative from the value, f (1 - f)
// and 1 - f^2. verify weighs the radius of each input dimension by the mean absolute entry of a
// column of this matrix (CORA's mean(abs(calcSensitivity(x)), 1)).
//
// Syntax:   J = nn.sensitivity(x);
// Inputs:   x - the point, a column (in, 1)
// Outputs:  J - the Jacobian (out, in)
// See also: cora/nn/neural_network/neural_network.h, evaluate.cpp
//
// CORA: nn/@neuralNetwork/calcSensitivity.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/nn/neural_network/neural_network.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor NeuralNetwork::sensitivity(const Tensor &x) const {
    detail::Entry entry;
    CORA_CHECK(x.shape() == std::vector<int64_t>({input_dim(), 1}), Error::dimension_mismatch, [&] {
        return "sensitivity: the point must be a column (" + std::to_string(input_dim()) +
               ", 1), got the shape " + shape_text(x.shape());
    });
    // the Jacobian of the layers so far, from the identity, and the value after them
    Tensor h = x, J = x.eye_like(input_dim());
    for (std::size_t k = 0; k < layers_.size(); ++k) {
        // the pre-activation of the layer
        const Tensor z = layers_[k].W.matmul(h) + layers_[k].b;
        // the value after the activation and its derivative
        const Tensor one = z.ones_like(), zero = z.zeros_like();
        Tensor derivative = one;
        switch (activations_[k]) {
        case Activation::identity:
            h = z;
            break;
        case Activation::relu:
            h = z.maximum(0.0);
            derivative = Tensor::where(z.greater(zero), one, zero);
            break;
        case Activation::sigmoid:
            h = z.sigmoid();
            derivative = h.multiply(one - h);
            break;
        case Activation::tanh:
            h = z.tanh();
            derivative = one - h.multiply(h);
            break;
        }
        // the chain rule: diag(a'(z)) W J
        J = derivative.multiply(layers_[k].W.matmul(J));
    }
    return J;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
