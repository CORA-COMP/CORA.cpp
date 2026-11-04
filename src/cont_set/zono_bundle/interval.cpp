// interval - the box around a zonotope bundle, as CORA's zonoBundle.interval
//
// Syntax:   Interval I = zB.interval();
// Inputs:   -
// Outputs:  I - the intersection of the boxes around the members: an enclosure of the bundle, in
//           general not the smallest box (empty if the boxes do not meet)
// See also: Zonotope::interval, con_zonotope
//
// CORA: contSet/@zonoBundle/interval.m

#include "cora/cont_set/zono_bundle/zono_bundle.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval ZonoBundle::interval_impl() const {
    Interval I = Z_.front().interval();
    for (std::size_t i = 1; i < Z_.size(); ++i) I = I.intersect(Z_[i].interval());
    return I;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
