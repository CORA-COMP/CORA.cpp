// volume - the volume of the empty set, as CORA's emptySet.volume_
//
// Syntax:   double v = E.volume();
// Inputs:   -
// Outputs:  v - 0, the Lebesgue measure of the empty set
// See also: radius, Fullspace::volume
//
// CORA: contSet/@emptySet/volume_.m

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
