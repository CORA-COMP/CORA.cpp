// priv_norm - the Euclidean length of columns
//
// Syntax:   Tensor len = priv_norm(x);
// Inputs:   x - columns (..., n, N)
// Outputs:  len - (..., 1, N): the root of the sum of squares, summed from the last row up as
//           CORALean's sqSumUp
// See also: priv_axis_distance
//
// CORA.cpp only: MATLAB calls norm and vecnorm

#include "cont_set/capsule/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor priv_norm(const Tensor &x) { return x.vector_norm(Tensor::Norm::two, -2, true); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
