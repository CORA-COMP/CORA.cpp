// polytope - the whole space as a polytope, as CORA's fullspace.polytope
//
// Syntax:   Polytope P(F);   // CORA: P = polytope(F)
// Inputs:   F - the whole space of n dimensions
// Outputs:  P - Polytope::inf(n)
// See also: Polytope::inf
//
// CORA: contSet/@fullspace/polytope.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/polytope/polytope.h"
#include "cora/cont_set/fullspace/fullspace.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Polytope::Polytope(const Fullspace &F) : Polytope(Polytope::inf(F.dim())) {}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
