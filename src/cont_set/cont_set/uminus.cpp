// uminus - the negated set, as CORA's contSet.uminus
//
// A class with its own negation (a box) overrides it.
//
// Syntax:   R = S.uminus();
// Inputs:   S - a set of dimension n
// Outputs:  R - {-x | x in S}, i.e. -1 * S, in the class of S
// See also: uplus, times, linear_map
//
// CORA: contSet/@contSet/uminus.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set ContSet::uminus() const {
    detail::Entry entry;
    return uminus_impl();
}

Set ContSet::uminus_impl() const { return linear_map_impl(-center_impl().eye_like(dim_impl())); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
