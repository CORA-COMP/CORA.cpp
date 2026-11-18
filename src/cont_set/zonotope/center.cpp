// center - the center of a zonotope, as CORA's zonotope.center
//
// Syntax:   c = Z.center();
// Inputs:   -
// Outputs:  c - the column (..., n, 1) the zonotope is built around
// See also: generators, ContSet::center
//
// CORA: contSet/@zonotope/center.m
// CORA: matrixSet/@matZonotope/center.m
// CORA authors: Matthias Althoff, Mark Wetzlinger
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

// the generic center is a column, as every other set's
Tensor Zonotope::center_impl() const { return elem_rank_ == 1 ? c_.expand_dims(-1) : c_; }


} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
