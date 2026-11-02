// support_function - the support function of a box, as CORA's interval.supportFunc
//
// Syntax:   rho = I.support_function(d);
// Inputs:   d - direction, a column (..., n, 1)
// Outputs:  rho - max_{x in I} d'x = d'c + |d|'r, shape (..., 1, 1)
// See also: Zonotope::support_function
//
// CORA: contSet/@interval/supportFunc_.m

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Interval::support_function_impl(const Tensor &d) const {
    return d.transpose().matmul(center()) + d.abs().transpose().matmul(rad());
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
