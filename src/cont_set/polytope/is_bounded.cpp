// is_bounded - whether a polytope is bounded, as CORA's polytope.isBounded
//
// A vertex polytope is bounded unless it has an infinite vertex in one dimension. A halfspace set
// is bounded if it is empty or its support function is finite along the axes and along the
// diagonal -1 (the n + 1 directions of a simplex); a coordinate that no row restricts is free. Not
// a batch for halfspaces.
//
// Syntax:   bool res = P.is_bounded();
// Inputs:   -
// Outputs:  res - whether the set is bounded; the empty set is
// See also: Polytope::is_full_dim, Polytope::representsa
//
// CORA: contSet/@polytope/isBounded.m
// CORA authors: Viktor Kotsev, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool Polytope::is_bounded_impl() const { return priv_polytope_is_bounded(*this); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
