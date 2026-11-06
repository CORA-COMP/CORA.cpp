// is_bounded - whether the empty set is bounded, as CORA's emptySet.isBounded
//
// Syntax:   bool b = E.is_bounded();
// Inputs:   -
// Outputs:  b - true
// See also: is_full_dim, Interval::is_bounded
//
// CORA: contSet/@emptySet/isBounded.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool EmptySet::is_bounded() const {
    detail::Entry entry;
    return true;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
