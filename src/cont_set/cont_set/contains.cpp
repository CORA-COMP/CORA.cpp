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
    detail::Entry entry;
    CORA_CHECK(S1.dim() == S2.dim(), Error::dimension_mismatch, [&] {
        return "contains: the sets differ in dimension, " + S1.class_name() + " of " +
               std::to_string(S1.dim()) + " and " + S2.class_name() + " of " +
               std::to_string(S2.dim());
    });
    CORA_CHECK(tol >= 0, Error::wrong_value,
               [&] { return "contains: tol must be nonnegative, got " + std::to_string(tol); });
    return detail::dispatch(BinaryOp::contains, S1, S2, tol);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
