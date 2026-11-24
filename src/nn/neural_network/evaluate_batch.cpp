// evaluate_batch - a batch of zonotopes through the network
//
// CORA's evaluateZonotopeBatch: B zonotopes (centers (B, n), generators (B, q, n)) go through the
// layers together, each activation by its line over the neuron's bounds (poly_method "bounds" or
// "singh"), the errors in generators of their own (the first num_approx_err of each layer, in the
// approx_error_order; the others in the radius dr of the interval center). The propagation is that
// of verify (priv_zono_batch.cpp).
//
// Syntax:   evaluate_batch() - the member of NeuralNetwork, e.g.
//           ZonotopeBatch Y = nn.evaluate_zonotope_batch(c, G);
// Inputs:   c - (B, n)
//           G - (B, q, n)
//           options - ZonotopeBatchOptions
// Outputs:  Y - the centers (B, m), generators (B, q + errors, m) and radius dr (B, m)
// See also: cora/nn/neural_network/neural_network.h, priv_zono_batch.h
// References:
//    [1] L. Koller, T. Ladner, M. Althoff. "Set-based training for neural network verification",
//        Transactions on Machine Learning Research, 2025.
//
// CORA: nn/@neuralNetwork/evaluateZonotopeBatch.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/error.h"
#include "cora/nn/neural_network/neural_network.h"
#include "global/format.h"
#include "nn/neural_network/private/priv_zono_batch.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

ZonotopeBatch NeuralNetwork::evaluate_zonotope_batch(const Tensor &c, const Tensor &G,
                                                     const ZonotopeBatchOptions &options) const {
    detail::Entry entry;
    CORA_CHECK(c.ndim() == 2 && G.ndim() == 3 && c.shape()[1] == input_dim() &&
                   G.shape()[0] == c.shape()[0] && G.shape()[2] == input_dim(),
               Error::dimension_mismatch, [&] {
                   return "evaluate_zonotope_batch: the centers are (B, " +
                          std::to_string(input_dim()) + ") and the generators (B, q, " +
                          std::to_string(input_dim()) + "), got " + shape_text(c.shape()) +
                          " and " + shape_text(G.shape());
               });
    const ZonoNet zono = priv_batch_net(*this, G.shape()[1], options);
    const ZonoPropagation out = zono.forward(c, Tensor(), G, Tensor(), nullptr, false);
    return {out.c, out.G, out.dr};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
