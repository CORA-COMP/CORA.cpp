// vertices_3d - the vertices of the empty set in space, as CORA's emptySet.vertices_
//
// Syntax:   V = E.vertices_3d();
// Inputs:   -
// Outputs:  V - no list of points: the set has no vertex
// See also: vertices, Interval::vertices_3d
//
// CORA: contSet/@emptySet/vertices_.m

#include "cora/cont_set/empty_set/empty_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::vector<std::vector<Point3>> EmptySet::vertices_3d_impl() const { return {}; }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
