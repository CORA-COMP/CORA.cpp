// vertices_3d - the vertices of the whole space in space, as CORA's fullspace.vertices_
//
// Syntax:   V = F.vertices_3d();
// Inputs:   -
// Outputs:  V - the eight corners of the unbounded box (+-Inf, +-Inf, +-Inf)
// See also: vertices, Interval::vertices_3d
//
// CORA: contSet/@fullspace/vertices_.m

#include "cora/cont_set/fullspace/fullspace.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::vector<std::vector<Point3>> Fullspace::vertices_3d_impl() const {
    return interval_impl().vertices_3d();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
