// is_bounded - whether a capsule is bounded
//
// Syntax:   res = C.is_bounded();
// Inputs:   -
// Outputs:  res - always true: a capsule is the ball of radius r around a segment
// See also: is_full_dim, Zonotope::is_bounded
//
// CORA: contSet/@capsule/isBounded.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/capsule/capsule.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool Capsule::is_bounded_impl() const { return true; }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
