// vertices - the vertices of the empty set, as CORA's emptySet.vertices_
//
// Syntax:   V = E.vertices();
// Inputs:   -
// Outputs:  V - no polygon: the set has no vertex (CORA: zeros(n, 0))
// See also: vertices_3d, Interval::vertices
//
// CORA: contSet/@emptySet/vertices_.m

#include "cora/cont_set/empty_set/empty_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::vector<Polygon> EmptySet::vertices_impl() const { return {}; }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
