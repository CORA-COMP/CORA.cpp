// priv_split_generator - the two halves of a zonotope along one generator
//
// Their union is Z (CORA's aux_splitOneDim).
//
// Syntax:   auto [lower, upper] = priv_split_generator(Z, j);
// Inputs:   Z - zonotope (a batch splits member by member)
//           j - the generator, 0 <= j < m
// Outputs:  lower, upper - the centers c - g_j/2 and c + g_j/2, with g_j halved in both
// See also: Zonotope::split, Zonotope::split_first_gen
//
// CORA: contSet/@zonotope/split.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cont_set/zonotope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::pair<Zonotope, Zonotope> priv_split_generator(const Zonotope &Z, int64_t j) {
    const Tensor &G = Z.generators();
    const int64_t m = G.size(-2);
    const Tensor half = G.slice(-2, j, j + 1) * 0.5;
    std::vector<Tensor> parts;
    if (j > 0) parts.push_back(G.slice(-2, 0, j));
    parts.push_back(half);
    if (j + 1 < m) parts.push_back(G.slice(-2, j + 1, m));
    const Tensor halved = Tensor::concat(parts, -2), shift = half.squeeze(-2);
    return {{Z.center() - shift, halved}, {Z.center() + shift, halved}};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
