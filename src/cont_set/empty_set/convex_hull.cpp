// convex_hull - the convex hull of the empty set and a set, as CORA's emptySet.convHull_
//
// Syntax:   Set R = E.convex_hull(S);
// Inputs:   S - a set of the dimension of E
// Outputs:  R - S itself: the empty set adds no point
// See also: intersect, Interval::convex_hull
//
// CORA: contSet/@emptySet/convHull_.m

#include "cora/cont_set/empty_set/empty_set.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set EmptySet::convex_hull(const Set &S) const {
    detail::Entry entry;
    CORA_CHECK(S->dim() == n_, Error::dimension_mismatch, [&] {
        return "convex_hull: the sets differ in dimension, " + std::to_string(n_) + " and " +
               std::to_string(S->dim());
    });
    return S;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
