// interval - the interval hull of the empty set, as CORA's emptySet.interval
//
// Syntax:   Interval I = E.interval();
// Inputs:   -
// Outputs:  I - the empty interval of the dimension of the set
// See also: Interval::empty
//
// CORA: contSet/@emptySet/interval.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval EmptySet::interval_impl() const { return Interval::empty(n_); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
