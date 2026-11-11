// center - the center of a polynomial zonotope, as CORA's polyZonotope.center
//
// The center c of the representation, not the center of the set (CORA's center).
//
// Syntax:   p = pZ.center();
// Inputs:   -
// Outputs:  p - the column c (n, 1)
// See also: zonotope
//
// CORA: contSet/@polyZonotope/center.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor PolyZonotope::center_impl() const { return c_; }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
