// is_full_dim - whether a capsule has a nonempty interior
//
// CORA tests r > 0 and misses the second case; the empty capsule has no interior. A batch: every
// member.
//
// Syntax:   res = C.is_full_dim();   res = C.is_full_dim(tol);
// Inputs:   tol - checked, relaxes nothing (CORA's isFullDim has none)
// Outputs:  res - true if r > 0, or r = 0 and the dimension is 1 with g != 0
// See also: is_bounded
//
// CORA: contSet/@capsule/isFullDim.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/capsule/private/priv.h"
#include "cora/cont_set/capsule/capsule.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool Capsule::is_full_dim_impl(Tol) const {
    const Tensor zero = r_.zeros_like();
    const Tensor ball = r_.greater(zero);
    if (dim() != 1) return priv_all(ball);
    // on the line a segment of positive length is an interval
    const Tensor segment = r_.greater_equal(zero).multiply(g_.abs().greater(zero));
    return priv_all(ball.maximum(segment));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
