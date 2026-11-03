// priv_within_tol - within_tol elementwise on tensors, as CORA's withinTol
//
// Syntax:   Tensor same = priv_within_tol(a, b, tol);
// Inputs:   a, b - tensors of one dtype (broadcast);  tol - nonnegative
// Outputs:  same - 1 where |a - b| <= tol, or |a - b| / min(|a|, |b|) <= tol, or a and b are the
//           same infinity, else 0 (NaN never agrees)
// See also: within_tol (global/within_tol.h), isequal, is_intersecting

#pragma once

#include "cora/global/tensor/tensor.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

Tensor priv_within_tol(const Tensor &a, const Tensor &b, double tol);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
