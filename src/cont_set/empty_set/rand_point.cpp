// rand_point - random points of the empty set, as CORA's emptySet.randPoint_
//
// Syntax:   P = E.rand_point(N, rng);
// Inputs:   N - number of points (ignored)
//           rng - random numbers (not used)
// Outputs:  P - the empty matrix (n, 0): the set has no point to draw
// See also: center, Interval::rand_point
//
// CORA: contSet/@emptySet/randPoint_.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

CORA_FROM_NUMBERS Tensor EmptySet::rand_point_impl(int64_t, Rng &) const {
    return Tensor::zeros({n_, 0});
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
