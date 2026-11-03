// center - the center of the whole space, as CORA's fullspace.center
//
// Syntax:   Tensor c = F.center();
// Inputs:   -
// Outputs:  c - the origin, a column (n, 1). CORA returns NaN in R^0, as MATLAB has no vector
//           of length 0 there; CORA.cpp returns the empty column (0, 1), as CORALean does
// See also: interval
//
// CORA: contSet/@fullspace/center.m

#include "cora/cont_set/fullspace/fullspace.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

CORA_FROM_NUMBERS Tensor Fullspace::center_impl() const { return Tensor::zeros({n_, 1}); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
