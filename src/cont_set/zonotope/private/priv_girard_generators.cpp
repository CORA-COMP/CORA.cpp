// priv_girard_generators - the generators of a zonotope after Girard's order reduction
//
// Syntax:   Tensor Gred = priv_girard_generators(G, keep);
// Inputs:   G - generators (..., r, m);  keep - how many stay, 0 <= keep <= m
// Outputs:  Gred - the keep generators of the largest Girard score ||g||_1 - ||g||_inf, then one
//           axis-aligned generator per row that bounds the rest (a batch reduces member by member)
// See also: Zonotope::reduce, ConZonotope::reduce
//
// CORA: contSet/@zonotope/private/priv_reduceGirard.m

#include "cont_set/zonotope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor priv_girard_generators(const Tensor &G, int64_t keep) {
    // the choice is made on detached values; the result takes its columns from G, with gradients
    const auto [kept, rest] = G.detach().girard_split(keep);
    const Tensor box = G.take_along_axis(rest, -1).abs_sum().diag();
    return keep == 0 ? box : Tensor::concat({G.take_along_axis(kept, -1), box}, -1);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
