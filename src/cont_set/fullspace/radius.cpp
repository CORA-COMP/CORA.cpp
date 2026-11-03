// radius - the radius of the smallest ball around the whole space, as CORA's fullspace.radius
//
// Syntax:   double r = F.radius();
// Inputs:   -
// Outputs:  r - Inf, the supremum of the norms over R^n; NaN in R^0, as CORA (CORALean has Inf
//           there and claims nothing)
// See also: volume, EmptySet::radius
//
// CORA: contSet/@fullspace/radius.m

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
