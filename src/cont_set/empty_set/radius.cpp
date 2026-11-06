// radius - the radius of the smallest ball around the empty set, as CORA's emptySet.radius
//
// Syntax:   double r = E.radius();
// Inputs:   -
// Outputs:  r - 0, the supremum of the norms over no point
// See also: volume, Interval::radius
//
// CORA: contSet/@emptySet/radius.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

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
