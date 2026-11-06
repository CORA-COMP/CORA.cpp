// is_intersecting - whether the whole space meets a set, as CORA's fullspace.isIntersecting_
//
// Syntax:   bool b = F.is_intersecting(S);   bool b = F.is_intersecting(S, tol);
// Inputs:   S - a set of the dimension of F (points p: the interval [p, p])
//           tol - checked and otherwise unused; default: the repository tolerance
//           (docs/architecture.md)
// Outputs:  b - true unless S is empty (an EmptySet or an interval without columns);
//           Error::not_supported in R^0. CORA's type argument has no effect and is not offered
// See also: contains, Interval::is_intersecting
//
// CORA: contSet/@fullspace/isIntersecting_.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/fullspace/fullspace.h"

#include "cont_set/fullspace/private/priv_represents_empty.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool Fullspace::is_intersecting(const ContSet &S, Tol tol) const {
    detail::Entry entry;
    CORA_CHECK(n_ > 0, Error::not_supported, [] {
        return std::string("is_intersecting: the intersection check of R^0 is not supported");
    });
    CORA_CHECK(S.dim() == n_, Error::dimension_mismatch, [&] {
        return "is_intersecting: the sets differ in dimension, " + std::to_string(n_) + " and " +
               std::to_string(S.dim());
    });
    static_cast<void>(resolve_tol(entry, tol, "is_intersecting"));
    return !priv_represents_empty(S);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
