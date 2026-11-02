// convex_hull - the convex hull of two sets, enclosed, as CORA's contSet.convHull
//
// Syntax:   S = convex_hull(S1, S2);
// Inputs:   S1, S2 - sets of one dimension
// Outputs:  S - a set that contains the convex hull of S1 ∪ S2
// See also: registry, unite
//
// CORA: contSet/@contSet/convHull.m

#include "cora/cont_set/registry.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set convex_hull(const ContSet &S1, const ContSet &S2) {
    detail::Entry entry;
    CORA_CHECK(S1.dim() == S2.dim(), Error::dimension_mismatch, [&] {
        return "convex_hull: the sets differ in dimension, " + S1.class_name() + " of " +
               std::to_string(S1.dim()) + " and " + S2.class_name() + " of " +
               std::to_string(S2.dim());
    });
    return detail::dispatch(BinaryOp::convex_hull, S1, S2);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
