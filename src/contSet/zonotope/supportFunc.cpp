// supportFunc - the support function of a zonotope, as CORA's zonotope.supportFunc
//
// Syntax:   rho = Z.supportFunc(d);
// Inputs:   d - direction, a column (..., n, 1)
// Outputs:  rho - max_{x in Z} d'x = d'c + sum_j |d'G_j|, shape (..., 1, 1)
// See also: Interval::supportFunc, Specification::check

#include "contSet/zonotope/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::ct {


// ===========================================  MAIN  =========================================== //

Tensor Zonotope::supportFunc(const Tensor &d) const {
    const Tensor dt = d.transpose();
    return dt.matmul(c) + dt.matmul(G).abs().sumLast();
}

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
