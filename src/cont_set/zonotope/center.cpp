// center - the center of a zonotope, as CORA's zonotope.center
//
// Syntax:   c = Z.center();
// Inputs:   -
// Outputs:  c - the point (..., n) the zonotope is built around (a matrix zonotope: (..., r, c))
// See also: generators, ContSet::center
//
// CORA: contSet/@zonotope/center.m
// CORA: matrixSet/@matZonotope/center.m
// CORA authors: Matthias Althoff, Mark Wetzlinger
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

// the generic center is a column, as every other set's
Tensor Zonotope::center_impl() const { return c_; }


} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
