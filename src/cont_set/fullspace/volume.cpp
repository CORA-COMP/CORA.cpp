// volume - the volume of the whole space, as CORA's fullspace.volume_
//
// Syntax:   double v = F.volume();
// Inputs:   -
// Outputs:  v - Inf, the Lebesgue measure of R^n; NaN in R^0, as CORA (CORALean has Inf there and
//           claims nothing)
// See also: radius, EmptySet::volume
//
// CORA: contSet/@fullspace/volume_.m

#include "cora/cont_set/fullspace/fullspace.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

double Fullspace::volume() const {
    detail::Entry entry;
    return n_ == 0 ? std::numeric_limits<double>::quiet_NaN()
                   : std::numeric_limits<double>::infinity();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
