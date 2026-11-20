// support_function - the support function of a zonotope, as CORA's zonotope.supportFunc
//
// Syntax:   rho = Z.support_function(d);
// Inputs:   d - direction (..., n)
// Outputs:  rho - max_{x in Z} d'x = d'c + sum_j |d'G_j|, shape (...)
// See also: Interval::support_function, Specification::check
//
// CORA: contSet/@zonotope/supportFunc_.m
// CORA authors: Niklas Kochdumper
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Zonotope::support_function_impl(const Tensor &d) const {
    // c'd and sum_j |g_j' d| on CORA's column of the direction: (..., 1, 1)
    const Tensor column = d.expand_dims(-1);
    const Tensor spread = G_.matmul(column).abs_sum_rows();
    const Tensor value = c_.expand_dims(-2).matmul(column) + spread;
    Shape batch = value.shape();
    batch.pop_back();
    batch.pop_back();
    return value.reshape(batch); // one reshape: the hot path of the halfspace checks
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
