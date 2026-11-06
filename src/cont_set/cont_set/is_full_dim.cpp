// is_full_dim - whether a set has a nonempty interior, as CORA's contSet.isFullDim
//
// Syntax:   res = S.is_full_dim();   res = S.is_full_dim(tol);
// Inputs:   S - set
//           tol - relaxes the width or rank test of the class; default: the repository
//           tolerance (docs/architecture.md)
// Outputs:  res - true if the set is not contained in a lower-dimensional subspace
// See also: is_bounded, Interval::is_full_dim, Zonotope::is_full_dim
//
// CORA: contSet/@contSet/isFullDim.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/cont_set/private/priv_noops.h"
#include "cora/cont_set/cont_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool ContSet::is_full_dim(Tol tol) const {
    detail::Entry entry;
    static_cast<void>(resolve_tol(entry, tol, center_impl().dtype(), "is_full_dim"));
    return is_full_dim_impl(tol);
}

bool ContSet::is_full_dim_impl(Tol) const { priv_noops("is_full_dim", *this); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
