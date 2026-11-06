// isequal - whether two sets are the same, as CORA's contSet.isequal
//
// A class relaxes its own parts: the bounds of a box, the center and generators of a zonotope.
//
// Syntax:   res = isequal(S1, S2);   res = isequal(S1, S2, tol);
// Inputs:   S1, S2 - sets of one dimension
//           tol - what the class relaxes; default: the repository tolerance
// Outputs:  res - true if the two sets are equal; the class of lower CORA precedence decides
// See also: registry, eq, ne
//
// CORA: contSet/@contSet/isequal.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/registry.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool isequal(const ContSet &S1, const ContSet &S2, Tol tol) {
    return detail::dispatch(BinaryOp::isequal, S1, S2, tol);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
