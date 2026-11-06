// vertices - the vertices of the whole plane, as CORA's fullspace.vertices_
//
// As CORA takes the vertices of interval(fs). R^0 is refused (Error::not_supported), as CORA
// does.
//
// Syntax:   V = F.vertices();
// Inputs:   -
// Outputs:  V - the four corners of the unbounded box (+-Inf, +-Inf)
// See also: vertices_3d, Interval::vertices
//
// CORA: contSet/@fullspace/vertices_.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/fullspace/fullspace.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::vector<Polygon> Fullspace::vertices_impl() const { return interval_impl().vertices(); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
