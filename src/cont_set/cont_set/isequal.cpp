// isequal - whether two sets are the same, as CORA's contSet.isequal
//
// Syntax:   res = isequal(S1, S2);   res = isequal(S1, S2, tol);
// Inputs:   S1, S2 - sets of one dimension;  tol - what the class relaxes (the bounds of a box, the
//           center and generators of a zonotope); default: the repository tolerance
//           (docs/architecture.md)
// Outputs:  res - true if the two sets are equal; the class of lower CORA precedence decides
// See also: registry, eq, ne
//
// CORA: contSet/@contSet/isequal.m

#include "cora/cont_set/registry.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool isequal(const ContSet &S1, const ContSet &S2, Tol tol) {
    return detail::dispatch(BinaryOp::isequal, S1, S2, tol);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
