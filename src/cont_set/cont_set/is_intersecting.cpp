// is_intersecting - whether two sets intersect, as CORA's contSet.isIntersecting
//
// Syntax:   res = is_intersecting(S1, S2);   res = is_intersecting(S1, S2, tol);
// Inputs:   S1, S2 - sets of one dimension;  tol - CORA's default 1e-8
// Outputs:  res - true if S1 and S2 + tol * B_inf intersect (B_inf the unit box)
// See also: registry, contains
//
// CORA: contSet/@contSet/isIntersecting.m

#include "cora/cont_set/registry.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool is_intersecting(const ContSet &S1, const ContSet &S2, double tol) {
    detail::Entry entry;
    CORA_CHECK(S1.dim() == S2.dim(), Error::dimension_mismatch, [&] {
        return "is_intersecting: the sets differ in dimension, " + S1.class_name() + " of " +
               std::to_string(S1.dim()) + " and " + S2.class_name() + " of " +
               std::to_string(S2.dim());
    });
    CORA_CHECK(tol >= 0, Error::wrong_value,
               [&] { return "is_intersecting: tol must be nonnegative, got " + std::to_string(tol); });
    return detail::dispatch(BinaryOp::is_intersecting, S1, S2, tol);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
