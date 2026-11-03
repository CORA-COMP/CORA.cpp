// contains - whether a set contains another one, as CORA's contSet.contains
//
// Syntax:   res = contains(S1, S2);   res = contains(S1, S2, tol);
// Inputs:   S1, S2 - sets of one dimension;  tol - how far outside S2 may lie; default: the
//           repository tolerance (docs/architecture.md)
//           (docs/architecture.md)
// Outputs:  res - true if S2 is a subset of S1 + tol * B_inf (B_inf the unit box)
// See also: registry, is_intersecting, Interval::contains
//
// CORA: contSet/@contSet/contains.m

#include "cora/cont_set/registry.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool contains(const ContSet &S1, const ContSet &S2, Tol tol) {
    detail::Entry entry;
    const detail::Refusal r = detail::refusal(BinaryOp::contains, S1, S2, tol);
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    return detail::contains_impl(S1, S2, tol);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
