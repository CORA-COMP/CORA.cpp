// backprop_zonotope_batch - the gradient of a loss on a batch of output zonotopes at the input's
//
// CORA's backpropZonotopeBatch: the gradients (grad_c, grad_gen) of a loss at the output set of a
// batch (the center, every generator and the radius of the interval center) go back through the
// layers to the gradients at the input's center and generators. Each linear layer maps them by W^T,
// each activation layer by the line of its neuron: with the lines held (CORA without
// exact_backprop) the slope scales the gradients and the error generators' gradients stop; exact
// (CORA's exact_backprop) the gradients also flow through the line's dependence on the neuron's
// bounds (slope, offset and error), as through the bounds' center and generators. CORA stores the
// forward pass in the layers; here the batch goes forward again (evaluate_zonotope_batch's
// propagation) and the weights are not updated (the network is immutable).
//
// Syntax:   backprop_zonotope_batch() - the member of NeuralNetwork, e.g.
//           ZonotopeBatch g = nn.backprop_zonotope_batch(c, G, grad_c, grad_gen);
// Inputs:   c, G - the input sets (B, n), (B, q, n)
//           grad_c - the gradient at the output's centers (B, m)
//           grad_gen - the gradient at the output's generators (B, q + errors, m)
//           grad_dr - the gradient at the interval center's radius (B, m), if any
//           options - ZonotopeBatchOptions of the forward pass
//           exact - through the lines' bounds too
// Outputs:  g - the gradients at the input: c (B, n) and G (B, q, n)
// See also: cora/nn/neural_network/neural_network.h, evaluate_batch.cpp, priv_zono_batch.h
// References:
//    [1] L. Koller, T. Ladner, M. Althoff. "Set-based training for neural network verification",
//        Transactions on Machine Learning Research, 2025.
//
// CORA: nn/@neuralNetwork/backpropZonotopeBatch.m
// CORA: nn/@neuralNetwork/backpropZonotopeBatch_.m
// CORA: nn/layers/nonlinear/@nnActivationLayer/backpropZonotopeBatch.m
// CORA authors: Lukas Koller
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/error.h"
#include "cora/nn/neural_network/neural_network.h"
#include "global/format.h"
#include "nn/neural_network/private/priv_zono_batch.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

ZonotopeBatch NeuralNetwork::backprop_zonotope_batch(const Tensor &c, const Tensor &G,
                                                     const Tensor &grad_c, const Tensor &grad_gen,
                                                     const Tensor &grad_dr,
                                                     const ZonotopeBatchOptions &options,
                                                     bool exact) const {
    detail::Entry entry;
    CORA_CHECK(
        c.ndim() == 2 && G.ndim() == 3 && c.shape()[1] == input_dim() &&
            G.shape()[2] == input_dim() && G.shape()[0] == c.shape()[0],
        Error::dimension_mismatch, [&] {
            return "backprop_zonotope_batch: the input sets are (B, n) and (B, q, n) with n = " +
                   std::to_string(input_dim()) + ", not " + shape_text(c.shape()) + " and " +
                   shape_text(G.shape());
        });
    // the forward pass, recorded
    const ZonoNet zono = priv_batch_net(*this, G.shape()[1], options);
    const ZonoPropagation out = zono.forward(c, Tensor(), G, Tensor(), nullptr, true);
    // the gradients are those of the output's center, generators and radius
    CORA_CHECK(grad_c.shape() == out.c.shape() && grad_gen.shape() == out.G.shape() &&
                   (!grad_dr.defined() || (out.dr.defined() && grad_dr.shape() == out.dr.shape())),
               Error::dimension_mismatch, [&] {
                   return "backprop_zonotope_batch: the output's gradients are " +
                          shape_text(out.c.shape()) + " and " + shape_text(out.G.shape()) +
                          " (and the radius's, if an error has no generator), got " +
                          shape_text(grad_c.shape()) + " and " + shape_text(grad_gen.shape());
               });
    // a radius without a gradient has gradient 0
    const Tensor g_dr =
        out.dr.defined() ? (grad_dr.defined() ? grad_dr : out.dr.zeros_like()) : Tensor();
    return priv_backprop_set(out, zono, grad_c, grad_gen, g_dr, exact);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
