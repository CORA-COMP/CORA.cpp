// center - the center of the empty set, as CORA's emptySet.center
//
// Syntax:   Tensor c = E.center();
// Inputs:   -
// Outputs:  c - the empty matrix (n, 0): the set has no point to be the center
// See also: Interval::center, rand_point
//
// CORA: contSet/@emptySet/center.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

CORA_FROM_NUMBERS Tensor EmptySet::center_impl() const { return Tensor::zeros({n_, 0}); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
