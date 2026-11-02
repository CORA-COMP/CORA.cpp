// unite - the union of two sets, enclosed, as CORA's contSet.or
//
// Syntax:   S = unite(S1, S2);   S = Set(S1) | Set(S2);
// Inputs:   S1, S2 - sets of one dimension
// Outputs:  S - a set that contains S1 ∪ S2
// See also: registry, intersect, convex_hull
//
// CORA: contSet/@contSet/or.m

#include "cora/cont_set/registry.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set unite(const ContSet &S1, const ContSet &S2) {
    return detail::dispatch(BinaryOp::unite, S1, S2);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
