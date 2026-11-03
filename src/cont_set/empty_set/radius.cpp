// radius - the radius of the smallest ball around the empty set, as CORA's emptySet.radius
//
// Syntax:   double r = E.radius();
// Inputs:   -
// Outputs:  r - 0, the supremum of the norms over no point
// See also: volume, Interval::radius
//
// CORA: contSet/@emptySet/radius.m

#include "cora/cont_set/empty_set/empty_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

double EmptySet::radius() const {
    detail::Entry entry;
    return 0.0;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
