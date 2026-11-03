// contains - whether the empty set contains a set, as CORA's emptySet.contains_
//
// Syntax:   bool b = E.contains(S);   bool b = E.contains(S, tol);   bool b = E.contains(p);
// Inputs:   S - a set of the dimension of E;  p - points (n, N)
//           tol - nonnegative; CORA passes it on to the emptiness test, default 100*eps
// Outputs:  b - true if S is empty, and for points if there are none (no columns)
// See also: isequal, Interval::contains
//
// CORA: contSet/@emptySet/contains_.m

#include "cont_set/empty_set/private/priv_is_empty_set.h"
#include "cora/cont_set/empty_set/empty_set.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool EmptySet::contains(const ContSet &S, double tol) const {
    detail::Entry entry;
    CORA_CHECK(S.dim() == n_, Error::dimension_mismatch, [&] {
        return "contains: the sets differ in dimension, " + std::to_string(n_) + " and " +
               std::to_string(S.dim());
    });
    CORA_CHECK(tol >= 0, Error::wrong_value,
               [&] { return "contains: tol must be nonnegative, got " + std::to_string(tol); });
    return priv_is_empty_set(S);
}

bool EmptySet::contains(const Tensor &p, double tol) const {
    detail::Entry entry;
    CORA_CHECK(p.ndim() >= 2 && p.size(-2) == n_, Error::dimension_mismatch, [&] {
        return "contains: the points must be a matrix with " + std::to_string(n_) +
               " rows, got the shape " + shape_text(p.shape());
    });
    CORA_CHECK(tol >= 0, Error::wrong_value,
               [&] { return "contains: tol must be nonnegative, got " + std::to_string(tol); });
    return p.size(-1) == 0;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
