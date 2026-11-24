// sensitivity - the Jacobian of a network at a point
//
// The chain rule through the layers in reverse: a forward pass computes z_k = W_k h_{k-1} + b_k,
// then J = I (out) becomes J diag(a_k'(z_k)) W_k for k = K, ..., 1, the order of CORA's
// calcSensitivity (a backward pass) and of CORALean's IntervalChain.sensitivity. A ReLU has the
// derivative 1 where its input is positive and 0 elsewhere (CORA's getDf(1) of
// nnLeakyReLULayer), sigmoid and tanh the derivative from the value, f (1 - f) and 1 - f^2. verify
// weighs the radius of each input dimension by the mean absolute entry of a column of this matrix
// (CORA's mean(abs(calcSensitivity(x)), 1)).
//
// Syntax:   J = nn.sensitivity(x);
// Inputs:   x - the point (in)
// Outputs:  J - the Jacobian (out, in)
// See also: cora/nn/neural_network/neural_network.h, evaluate.cpp
// References:
//    [1] J. M. Zurada, A. Malinowski, I. Cloete. "Sensitivity analysis for minimization of input
//        data dimension for feedforward neural network", ISCAS 1994, vol. 6, pp. 447-450.
//
// CORA: nn/@neuralNetwork/calcSensitivity.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       11-November-2026
// Last update:   24-November-2026
// Reviewed:      ---

#include "cora/nn/neural_network/neural_network.h"
#include "global/format.h"
#include "nn/neural_network/private/priv_layers.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

// The Jacobian ------------------------------------------------------------------------------------

Tensor NeuralNetwork::sensitivity(const Tensor &x) const {
    detail::Entry entry;
    CORA_CHECK(x.shape() == std::vector<int64_t>({input_dim()}), Error::dimension_mismatch, [&] {
        return "sensitivity: the point must be a vector (" + std::to_string(input_dim()) +
               "), got the shape " + shape_text(x.shape());
    });
    CORA_CHECK(is_dense() && is_chain(), Error::not_supported, [] {
        return std::string("sensitivity: the layers must be a chain of matrices (Layer::dense)");
    });
    // the forward pass: each layer's pre-activation z_k and the derivative of its activation there
    std::vector<Tensor> slopes;
    slopes.reserve(layers_.size());
    Tensor h = x.expand_dims(-1);
    for (std::size_t k = 0; k < layers_.size(); ++k) {
        const Tensor z = layers_[k].W.matmul(h) + layers_[k].b.expand_dims(-1);
        const PrivDerivative d = priv_activation_derivative(activations_[k], z);
        h = d.value;
        slopes.push_back(d.derivative.matrix_transpose());
    }
    // The reverse pass ----------------------------------------------------------------------------
    // from the output back, J diag(a_k'(z_k)) W_k (CORALean's IntervalChain.sensitivity)
    Tensor J = x.eye_like(output_dim());
    for (std::size_t k = layers_.size(); k-- > 0;) J = J.multiply(slopes[k]).matmul(layers_[k].W);
    return J;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
