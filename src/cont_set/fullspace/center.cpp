// center - the center of the whole space, as CORA's fullspace.center
//
// CORA returns NaN in R^0, as MATLAB has no vector of length 0 there; CORA.cpp returns the empty
// column (0, 1), as CORALean does.
//
// Syntax:   Tensor c = F.center();
// Inputs:   -
// Outputs:  c - the origin, a column (n, 1)
// See also: interval
//
// CORA: contSet/@fullspace/center.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/fullspace/fullspace.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

CORA_FROM_NUMBERS Tensor Fullspace::center_impl() const { return Tensor::zeros({n_, 1}); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
