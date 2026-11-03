// within_tol - within_tol elementwise on tensors, as CORA's withinTol
//
// Syntax:   Tensor same = within_tol(a, b, tol);
// Inputs:   a, b - tensors of one dtype (broadcast);  tol - nonnegative, default: tolerance.h
// Outputs:  same - 1 where |a - b| <= tol, or |a - b| / min(|a|, |b|) <= tol, or a and b are the
//           same infinity, else 0 (NaN never agrees)
// See also: cora/global/within_tol.h
//
// CORA.cpp only: the tensor form of withinTol, which MATLAB gets by vectorization

#include "cora/global/within_tol.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor within_tol(const Tensor &a, const Tensor &b, Tol tol) {
    const Tensor difference = (a - b).abs();
    const Tensor bound =
        difference.full_like(difference.shape(), resolve_tol(tol, a.dtype()));
    const Tensor relative = difference.divide(a.abs().minimum(b.abs()));
    // equal values include the equal infinities, where the difference is NaN
    return difference.less_equal(bound).maximum(relative.less_equal(bound)).maximum(a.equal(b));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
