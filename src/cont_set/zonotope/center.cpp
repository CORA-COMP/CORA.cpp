// center - the center of a zonotope, as CORA's zonotope.center
//
// Syntax:   c = Z.center();
// Inputs:   -
// Outputs:  c - the column (..., n, 1) the zonotope is built around
// See also: generators, ContSet::center
//
// CORA: contSet/@zonotope/center.m
// CORA authors: Matthias Althoff, Mark Wetzlinger
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Zonotope::center_impl() const { return c_; }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
