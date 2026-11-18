// support_function - the support function of a zonotope, as CORA's zonotope.supportFunc
//
// Syntax:   rho = Z.support_function(d);
// Inputs:   d - direction, a column (..., n, 1)
// Outputs:  rho - max_{x in Z} d'x = d'c + sum_j |d'G_j|, shape (..., 1, 1)
// See also: Interval::support_function, Specification::check
//
// CORA: contSet/@zonotope/supportFunc_.m
// CORA authors: Niklas Kochdumper
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Zonotope::support_function_impl(const Tensor &d) const {
    // c'd and sum_j |g_j' d|, one row per direction: (..., k, 1)
    const Tensor spread = G_.matmul(d).abs_sum_rows();
    if (d.size(-1) == 1) return c_.expand_dims(-2).matmul(d) + spread;
    return d.matrix_transpose().matmul(c_.expand_dims(-1)) + spread;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
