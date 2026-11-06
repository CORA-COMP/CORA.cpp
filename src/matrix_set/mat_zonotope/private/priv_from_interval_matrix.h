// priv_from_interval_matrix - an interval matrix as a matrix zonotope
//
// CORA's intervalMatrix/matZonotope.
//
// Syntax:   MatZonotope Z = priv_from_interval_matrix(A);
// Inputs:   A - interval matrix (..., n, m)
// Outputs:  Z - center of A and one generator per entry, the radius of that entry, in column order
// See also: expm_independent, interval_matrix
//
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/matrix_set/mat_zonotope/mat_zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

MatZonotope priv_from_interval_matrix(const IntervalMatrix &A);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
