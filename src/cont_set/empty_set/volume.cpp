// volume - the volume of the empty set, as CORA's emptySet.volume_
//
// Syntax:   double v = E.volume();
// Inputs:   -
// Outputs:  v - 0, the Lebesgue measure of the empty set
// See also: radius, Fullspace::volume
//
// CORA: contSet/@emptySet/volume_.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

double EmptySet::volume() const {
    detail::Entry entry;
    return 0.0;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
