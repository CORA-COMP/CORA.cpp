// generator_reduction - CORA's generator reduction layer on a batch of zonotopes
//
// A batch of zonotopes (B, q, n) with more than max_gens generators is reduced to exactly max_gens:
// the generators are sorted by their L1 length (ascending, stable), the first max_gens - n stay as
// they are, and the others are replaced by the box that contains them, one axis-parallel
// generator per dimension of the radius sum |g| of the replaced ones. The enclosure is sound
// (the box contains the sum of the replaced generators' segments). CORA keeps the shortest
// generators and boxes the longest, the opposite of Girard's reduction; this port follows CORA.
// A batch whose generators fit (q <= max_gens - n) is returned unchanged, where CORA's indexing
// would fail. The center and the radius dr of the interval center are untouched.
//
// Syntax:   generator_reduction() - the reduction, e.g.
//           ZonotopeBatch R = NeuralNetwork::reduce_generators(Z, max_gens);
// Inputs:   Z - the batch (c (B, n), G (B, q, n), dr (B, n) or undefined); max_gens >= n
// Outputs:  R - the reduced batch with max_gens generators
// See also: cora/nn/neural_network/neural_network.h, evaluate_batch.cpp
// References: none (CORA's layer, no paper's algorithm)
//
// CORA: nn/layers/linear/nnGeneratorReductionLayer.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/error.h"
#include "cora/nn/neural_network/neural_network.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

ZonotopeBatch NeuralNetwork::reduce_generators(const ZonotopeBatch &Z, int64_t max_gens) {
    detail::Entry entry;
    CORA_CHECK(Z.c.ndim() == 2 && Z.G.ndim() == 3 && Z.G.shape()[0] == Z.c.shape()[0] &&
                   Z.G.shape()[2] == Z.c.shape()[1],
               Error::dimension_mismatch, [&] {
                   return "reduce_generators: the centers are (B, n) and the generators (B, q, "
                          "n), got " +
                          shape_text(Z.c.shape()) + " and " + shape_text(Z.G.shape());
               });
    const int64_t n = Z.c.shape()[1], q = Z.G.shape()[1];
    CORA_CHECK(max_gens >= n, Error::wrong_value, [&] {
        return "reduce_generators: max_gens must be at least the dimension " + std::to_string(n) +
               ", got " + std::to_string(max_gens);
    });
    const int64_t keep = max_gens - n;
    if (q <= keep) return Z;
    // the generators by ascending length: the first `keep` stay, the rest become a box
    const Tensor order = Z.G.abs().sum(-1).argsort(1);
    const Tensor rows = order.expand_dims(-1).broadcast_to({Z.G.shape()[0], q, n});
    const Tensor sorted = Z.G.take_along_axis(rows, 1);
    const Tensor box = sorted.slice(1, keep, q).abs().sum(1);
    const Tensor diagonal = box.expand_dims(-1).multiply(box.eye_like(n).expand_dims(0));
    return {Z.c, Tensor::concat({sorted.slice(1, 0, keep), diagonal}, 1), Z.dr};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
