// vertices_3d - the vertices of the whole space in space, as CORA's fullspace.vertices_
//
// Syntax:   V = F.vertices_3d();
// Inputs:   -
// Outputs:  V - the eight corners of the unbounded box (+-Inf, +-Inf, +-Inf)
// See also: vertices, Interval::vertices_3d
//
// CORA: contSet/@fullspace/vertices_.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#include "cora/cont_set/fullspace/fullspace.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::vector<std::vector<Point3>> Fullspace::vertices_3d_impl() const {
    return interval_impl().vertices_3d();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
