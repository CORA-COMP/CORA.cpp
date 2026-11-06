// fullspace - the constructor of the whole space, checked, as CORA's fullspace constructor
//
// Syntax:   Fullspace F(n);
// Inputs:   n - the dimension, an integer >= 0
// Outputs:  F - the set R^n, whose dim() is n (CORA's dim)
// See also: inf, interval
//
// CORA: contSet/@fullspace/dim.m
// CORA: contSet/@fullspace/fullspace.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/fullspace/fullspace.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Fullspace::Fullspace(int64_t n) : DimensionSet(kClassId, n) {}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
