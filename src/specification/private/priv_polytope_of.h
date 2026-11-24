// priv_polytope_of - the set of a specification as a polytope (CORA's polytope(S))
//
// Syntax:   Polytope P = priv_polytope_of(S);
// Inputs:   S - a set of a class with a polytope constructor, else Error::noops
// Outputs:  P - the same set as a polytope
// See also: Polytope
//
// CORA.cpp only: CORA's polytope constructor accepts a contSet and dispatches on its class
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/polytope/polytope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

Polytope priv_polytope_of(const Set &S);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
