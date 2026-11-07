// polytope - the empty set as a polytope, as CORA's emptySet.polytope
//
// Syntax:   Polytope P(E);   // CORA: P = polytope(E)
// Inputs:   E - the empty set of n dimensions
// Outputs:  P - Polytope::empty(n)
// See also: Polytope::empty
//
// CORA: contSet/@emptySet/polytope.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/polytope/polytope.h"
#include "cora/cont_set/empty_set/empty_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Polytope::Polytope(const EmptySet &E) : Polytope(Polytope::empty(E.dim())) {}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
