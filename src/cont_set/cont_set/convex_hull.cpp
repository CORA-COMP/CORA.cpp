// convex_hull - the convex hull of two sets, enclosed, as CORA's contSet.convHull
//
// Syntax:   S = convex_hull(S1, S2);
// Inputs:   S1, S2 - sets of one dimension
// Outputs:  S - a set that contains the convex hull of S1 ∪ S2
// See also: registry, unite
//
// CORA: contSet/@contSet/convHull.m
// CORA: contSet/@contSet/convHull_.m
// CORA authors: Mark Wetzlinger, Tobias Ladner
// Authors:       Tobias Ladner, Claude Sonnet 5.5, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/registry.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set convex_hull(const ContSet &S1, const ContSet &S2) {
    return detail::dispatch(BinaryOp::convex_hull, S1, S2);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
