// priv_from_interval_matrix - an interval matrix as a matrix zonotope
//
// CORA's intervalMatrix/matZonotope.
//
// Syntax:   Zonotope Z = priv_from_interval_matrix(A);
// Inputs:   A - interval matrix (..., n, m)
// Outputs:  Z - center of A and one generator per entry, the radius of that entry, in column order
// See also: expm_independent, interval_matrix
//
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_dynamics/linear_sys/mat_exp/mat_exp.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

Zonotope priv_from_interval_matrix(const Interval &I);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
