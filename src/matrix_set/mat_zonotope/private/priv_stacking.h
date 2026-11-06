// priv_stacking - the helpers of the matrix zonotope that reshape and join matrices
//
// Syntax:   Tensor v = priv_stack_columns(T);   Tensor G = priv_stack_generators(Gs);
//           Tensor J = priv_join(A, B);   Tensor Z = priv_no_generators(C);
// Inputs:   T - matrices (..., n, m)
//           Gs - generators (..., h, n, m)
//           A, B - generators to join;
//           C - the matrix whose shape and backend the result takes
// Outputs:  v - the matrices as columns (..., n m, 1), entry (i, j) at j n + i (MATLAB's T(:));
//           G - the generators as columns (..., n m, h);  J - A and B along the generator axis;
//           Z - the empty list of generators (..., 0, n, m)
// See also: zonotope, contains, mat_zonotope.h
//
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/global/tensor/tensor.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The shape of t with its last k axes replaced by `tail`.
Shape priv_replace_tail(const Tensor &t, int k, std::initializer_list<int64_t> tail);

Tensor priv_stack_columns(const Tensor &T);

Tensor priv_stack_generators(const Tensor &Gs);

Tensor priv_join(const Tensor &A, const Tensor &B);

Tensor priv_no_generators(const Tensor &C);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
