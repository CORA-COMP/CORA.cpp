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
    // c'd and sum_j |g_j' d| on the rows; one direction: (..., h, 1) is the memory of (..., 1, h)
    const Tensor at_center = c_.expand_dims(-2).matmul(d), images = G_.matmul(d);
    if (d.size(-1) == 1) return at_center + images.squeeze(-1).expand_dims(-2).abs_sum();
    return at_center.matrix_transpose() + images.matrix_transpose().abs_sum();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
