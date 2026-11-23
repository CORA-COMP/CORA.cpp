// is_bounded - whether a polynomial zonotope is bounded, as CORA's isBounded
//
// Always: the factors range over [-1, 1] and the generators are finite.
//
// Syntax:   bool res = pZ.is_bounded();
// Inputs:   -
// Outputs:  res - true
// See also: is_full_dim
//
// CORA: contSet/@polyZonotope/isBounded.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/poly_zonotope/poly_zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool PolyZonotope::is_bounded_impl() const { return true; }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
