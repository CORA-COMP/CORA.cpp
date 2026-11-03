// convex_hull - the convex hull of the whole space and a set, as CORA's fullspace.convHull_
//
// Syntax:   Fullspace H = F.convex_hull(S);
// Inputs:   S - a set of the dimension of F
// Outputs:  H - the whole space, which contains every set
// See also: Interval::convex_hull
//
// CORA: contSet/@fullspace/convHull_.m

#include "cora/cont_set/fullspace/fullspace.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Fullspace Fullspace::convex_hull(const ContSet &S) const {
    detail::Entry entry;
    CORA_CHECK(S.dim() == n_, Error::dimension_mismatch, [&] {
        return "convex_hull: the sets differ in dimension, " + std::to_string(n_) + " and " +
               std::to_string(S.dim());
    });
    return *this;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
