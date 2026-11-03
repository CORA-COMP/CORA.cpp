// priv_within_tol - within_tol of every entry of a tensor and a finite number, as a mask
//
// Syntax:   Tensor mask = priv_within_tol(x, b, tol);
// Inputs:   x - tensor;  b - finite number;  tol - nonnegative tolerance
// Outputs:  mask - 1 where |x - b| <= tol or |x - b| <= tol * min(|x|, |b|), else 0 (NaN: 0)
// See also: cora/global/within_tol.h
//
// CORA.cpp only: the tensor form of withinTol, which MATLAB gets by vectorization

#include "cont_set/zonotope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor priv_within_tol(const Tensor &x, double b, double tol) {
    // two equal infinities cannot occur: b is finite
    const Tensor difference = (x - b).abs();
    const Tensor absolute = difference.less_equal(difference.full_like(difference.shape(), tol));
    const Tensor relative =
        difference.less_equal(x.abs().minimum(std::abs(b)) * tol);
    return absolute.maximum(relative);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
