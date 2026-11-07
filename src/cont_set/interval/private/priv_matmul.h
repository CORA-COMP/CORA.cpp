// priv_matmul - the product of two interval matrices
//
// The products are of A(i, k) and B(k, j), a NaN corner (0 * Inf) taken as 0, as CORA's times and
// mtimes. The sum over k is nested to the right, a(0) + (a(1) + ...), as CORALean's sumDown.
//
// Syntax:   Interval P = priv_matmul(A, B);
// Inputs:   A - interval matrix (..., m, n)
//           B - interval matrix (..., n, p)
// Outputs:  P - entry (i, j) is the sum over k of the hull of the four corner products
// See also: matrix_power, linear_map
//
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

Interval priv_matmul(const Interval &A, const Interval &B);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
