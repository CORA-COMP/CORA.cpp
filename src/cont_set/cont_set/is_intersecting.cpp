// is_intersecting - whether two sets intersect, as CORA's contSet.isIntersecting
//
// Syntax:   res = is_intersecting(S1, S2);   res = is_intersecting(S1, S2, tol);
// Inputs:   S1, S2 - sets of one dimension;  tol - how far apart the sets may be; default: the
//           repository tolerance (docs/architecture.md)
//           (docs/architecture.md)
// Outputs:  res - true if S1 and S2 + tol * B_inf intersect (B_inf the unit box)
// See also: registry, contains
//
// CORA: contSet/@contSet/isIntersecting.m

#include "cora/cont_set/registry.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool is_intersecting(const ContSet &S1, const ContSet &S2, Tol tol) {
    detail::Entry entry;
    const detail::Refusal r = detail::refusal(BinaryOp::is_intersecting, S1, S2, tol);
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    return detail::is_intersecting_impl(S1, S2, tol);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
