// interval - the box around a zonotope bundle, as CORA's zonoBundle.interval
//
// The intersection of the boxes around the members: an enclosure, in general not the smallest box.
//
// Syntax:   Interval I = bundle.interval();
// Inputs:   -
// Outputs:  I - the box, empty if the boxes of the members do not meet
// See also: Zonotope::interval, con_zonotope
//
// CORA: contSet/@zonoBundle/interval.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

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
