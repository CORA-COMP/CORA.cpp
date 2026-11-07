// convex_hull - an ellipsoid that holds the convex hull of two, as CORA's ellipsoid.convHull_
//
// The union of two convex sets is enclosed by their hull's enclosure: unite in mode "outer". CORA
// returns nothing for an empty second operand; here it is the first ellipsoid.
//
// Syntax:   Eres = E.convex_hull(S);
// Inputs:   S - ellipsoid of the same dimension
// Outputs:  Eres - an ellipsoid that holds the hull
// See also: unite, Capsule::convex_hull
//
// CORA: contSet/@ellipsoid/convHull_.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Ellipsoid Ellipsoid::convex_hull(const Ellipsoid &S) const { return unite(S, "outer"); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
