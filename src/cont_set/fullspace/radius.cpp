// radius - the radius of the smallest ball around the whole space, as CORA's fullspace.radius
//
// NaN in R^0, as CORA (CORALean has Inf there and claims nothing).
//
// Syntax:   double r = F.radius();
// Inputs:   -
// Outputs:  r - Inf, the supremum of the norms over R^n
// See also: volume, EmptySet::radius
//
// CORA: contSet/@fullspace/radius.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/fullspace/fullspace.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

double Fullspace::radius() const {
    detail::Entry entry;
    return n_ == 0 ? std::numeric_limits<double>::quiet_NaN()
                   : std::numeric_limits<double>::infinity();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
