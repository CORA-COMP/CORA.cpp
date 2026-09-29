// mtimes - the linear map M * I of a box, as CORA's interval.mtimes
//
// Syntax:   Ires = I.mtimes(M);
// Inputs:   M - matrix (..., n, n)
// Outputs:  Ires - the interval hull of M * I: center M*c, radius |M|*r
// See also: Zonotope::mtimes

#include "contSet/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::ct {

// ===========================================  MAIN  =========================================== //

Interval Interval::mtimes(const Tensor &M) const {
    const Tensor c = M.matmul(center());
    const Tensor r = M.abs().matmul(rad());
    return {c - r, c + r};
}

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
