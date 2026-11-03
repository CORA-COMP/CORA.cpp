// priv_vec - the bounds of an interval matrix stacked into one column, as reshape(I, [], 1)
//
// Syntax:   Interval v = priv_vec(I);
// Inputs:   I - interval of matrix bounds (..., rows, cols)
// Outputs:  v - the interval of bounds (..., rows * cols, 1), entry (i, j) at position i + rows * j
// See also: representsa, vertices, volume

#pragma once

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

Interval priv_vec(const Interval &I);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
