// supportFunc - the support function of a box, as CORA's interval.supportFunc
//
// Syntax:   rho = I.supportFunc(d);
// Inputs:   d - direction, a column (..., n, 1)
// Outputs:  rho - max_{x in I} d'x = d'c + |d|'r, shape (..., 1, 1)
// See also: Zonotope::supportFunc

#include "contSet/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::ct {

// ===========================================  MAIN  =========================================== //

Tensor Interval::supportFunc(const Tensor &d) const {
    return d.transpose().matmul(center()) + d.abs().transpose().matmul(rad());
}

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
