// is_intersecting - whether a set meets the empty set, as CORA's emptySet.isIntersecting_
//
// Syntax:   bool b = E.is_intersecting(S);   bool b = E.is_intersecting(S, tol);
// Inputs:   S - a set of the dimension of E
//           tol - nonnegative, default: the repository tolerance; relaxes nothing here
// Outputs:  b - false: the empty set has no point in common with any set
// See also: intersect, Interval::is_intersecting
//
// CORA: contSet/@emptySet/isIntersecting_.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool EmptySet::is_intersecting(const ContSet &S, Tol tol_arg) const {
    detail::Entry entry;
    CORA_CHECK(S.dim() == n_, Error::dimension_mismatch, [&] {
        return "is_intersecting: the sets differ in dimension, " + std::to_string(n_) + " and " +
               std::to_string(S.dim());
    });
    resolve_tol(entry, tol_arg, S.center().dtype(), "is_intersecting"); // checks tol only
    return false;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
