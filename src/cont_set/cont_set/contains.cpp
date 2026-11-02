// contains - whether a set contains another one, as CORA's contSet.contains
//
// Syntax:   res = contains(S1, S2);   res = contains(S1, S2, tol);
// Inputs:   S1, S2 - sets of one dimension;  tol - CORA's default 100*eps
// Outputs:  res - true if S2 is a subset of S1 + tol * B_inf (B_inf the unit box)
// See also: registry, is_intersecting, Interval::contains
//
// CORA: contSet/@contSet/contains.m

#include "cora/cont_set/registry.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool contains(const ContSet &S1, const ContSet &S2, double tol) {
    return detail::dispatch(BinaryOp::contains, S1, S2, tol);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
