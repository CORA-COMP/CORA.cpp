// is_bounded - whether a set is bounded, as CORA's contSet.isBounded
//
// Syntax:   res = S.is_bounded();
// Inputs:   S - set
// Outputs:  res - true if the set lies in a ball; for a batch, if every member does
// See also: is_full_dim, Interval::is_bounded, Zonotope::is_bounded
//
// CORA: contSet/@contSet/isBounded.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/cont_set/private/priv_noops.h"
#include "cora/cont_set/cont_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool ContSet::is_bounded() const {
    detail::Entry entry;
    return is_bounded_impl();
}

bool ContSet::is_bounded_impl() const { priv_noops("is_bounded", *this); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
