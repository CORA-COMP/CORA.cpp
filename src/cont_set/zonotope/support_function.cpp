// support_function - the support function of a zonotope, as CORA's zonotope.supportFunc
//
// Syntax:   rho = Z.support_function(d);
// Inputs:   d - direction, a column (..., n, 1)
// Outputs:  rho - max_{x in Z} d'x = d'c + sum_j |d'G_j|, shape (..., 1, 1)
// See also: Interval::support_function, Specification::check
//
// CORA: contSet/@zonotope/supportFunc_.m

#include "cora/cont_set/zonotope/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Zonotope::support_function_impl(const Tensor &d) const {
    const Tensor dt = d.matrix_transpose();
    return dt.matmul(c_) + dt.matmul(G_).abs_sum();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
