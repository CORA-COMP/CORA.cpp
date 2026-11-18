// priv_girard_generators - the generators of a zonotope after Girard's order reduction
//
// The box is one axis-aligned generator per coordinate that bounds the rest. A batch reduces member
// by member.
//
// Syntax:   Tensor Gred = priv_girard_generators(G, keep);
// Inputs:   G - generators (..., m, r), one per row
//           keep - how many stay, 0 <= keep <= m
// Outputs:  Gred - the keep generators of the largest Girard score ||g||_1 - ||g||_inf, then a box
// See also: Zonotope::reduce, ConZonotope::reduce
//
// CORA: contSet/@zonotope/private/priv_reduceGirard.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cont_set/zonotope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor priv_girard_generators(const Tensor &G, int64_t keep) {
    // the choice is made on detached values; the result takes its rows from G, with gradients
    const auto [kept, rest] = G.detach().girard_split(keep);
    const Tensor box = G.take_along_axis(rest, -2).abs().sum(-2).expand_dims(-1).diag();
    return keep == 0 ? box : Tensor::concat({G.take_along_axis(kept, -2), box}, -2);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
