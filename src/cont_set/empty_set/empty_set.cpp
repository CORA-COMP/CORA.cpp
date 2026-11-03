// empty_set - the constructor of the empty set, checked, as CORA's emptySet constructor
//
// Syntax:   EmptySet E(n);
// Inputs:   n - the dimension, an integer >= 0
// Outputs:  E - the empty set of n dimensions
// See also: empty, Interval::Interval
//
// CORA: contSet/@emptySet/emptySet.m

#include "cora/cont_set/empty_set/empty_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

EmptySet::EmptySet(int64_t n) : DimensionSet(kClassId, n) {}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
