// priv_from_interval_matrix - an interval matrix as a matrix zonotope
//
// Syntax:   MatZonotope Z = priv_from_interval_matrix(A);
// Inputs:   A - interval matrix (..., n, m)
// Outputs:  Z - center of A and one generator per entry, the radius of that entry, in column
//           order (CORA's intervalMatrix/matZonotope)
// See also: expm_independent, interval_matrix

#pragma once

#include "cora/matrix_set/mat_zonotope/mat_zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

MatZonotope priv_from_interval_matrix(const IntervalMatrix &A);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
