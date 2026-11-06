// empty_set - the constructor of the empty set, checked, as CORA's emptySet constructor
//
// Syntax:   EmptySet E(n);
// Inputs:   n - the dimension, an integer >= 0
// Outputs:  E - the empty set of n dimensions, whose dim() is n (CORA's dim)
// See also: empty, Interval::Interval
//
// CORA: contSet/@emptySet/dim.m
// CORA: contSet/@emptySet/emptySet.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

EmptySet::EmptySet(int64_t n) : DimensionSet(kClassId, n) {}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
