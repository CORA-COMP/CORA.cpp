// priv_empty - the empty interval of the shape of a tensor
//
// Syntax:   Interval E = priv_empty(like);
// Inputs:   like - a tensor of the shape of the bounds (..., n, k) and the backend of the result
// Outputs:  E - the interval of bounds (..., n, 0): no columns, as CORA's empty interval
// See also: intersect, minkowski_difference

#pragma once

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

Interval priv_empty(const Tensor &like);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
