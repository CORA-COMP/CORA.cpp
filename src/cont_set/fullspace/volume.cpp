// volume - the volume of the whole space, as CORA's fullspace.volume_
//
// NaN in R^0, as CORA (CORALean has Inf there and claims nothing).
//
// Syntax:   double v = F.volume();
// Inputs:   -
// Outputs:  v - Inf, the Lebesgue measure of R^n
// See also: radius, EmptySet::volume
//
// CORA: contSet/@fullspace/volume_.m
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

double Fullspace::volume() const {
    detail::Entry entry;
    return n_ == 0 ? std::numeric_limits<double>::quiet_NaN()
                   : std::numeric_limits<double>::infinity();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
