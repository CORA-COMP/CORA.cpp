// contains - whether the empty set contains a set, as CORA's emptySet.contains_
//
// Syntax:   bool b = E.contains(S);   bool b = E.contains(S, tol);   bool b = E.contains(p);
// Inputs:   S - a set of the dimension of E
//           p - points (..., n), one per row
//           tol - nonnegative; CORA passes it on to the emptiness test; default: the repository one
// Outputs:  b - true if S is empty, and for points if there are none (no columns)
// See also: isequal, Interval::contains
//
// CORA: contSet/@emptySet/contains_.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/empty_set/private/priv_is_empty_set.h"
#include "cora/cont_set/empty_set/empty_set.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool EmptySet::contains(const ContSet &S, Tol tol_arg) const {
    detail::Entry entry;
    CORA_CHECK(S.dim() == n_, Error::dimension_mismatch, [&] {
        return "contains: the sets differ in dimension, " + std::to_string(n_) + " and " +
               std::to_string(S.dim());
    });
    resolve_tol(entry, tol_arg, S.center().dtype(), "contains"); // checks tol; relaxes nothing
    return priv_is_empty_set(S);
}

bool EmptySet::contains(const Tensor &p, Tol tol_arg) const {
    detail::Entry entry;
    CORA_CHECK(is_vector(p, n_), Error::dimension_mismatch, [&] {
        return "contains: the points must be (..., " + std::to_string(n_) +
               "), one per row, got the shape " + shape_text(p.shape());
    });
    resolve_tol(entry, tol_arg, p.dtype(), "contains"); // checks tol; relaxes nothing
    // only no point at all, (..., 0, n), is in the empty set
    return p.numel() == 0;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
