// priv_within_tol - within_tol elementwise on tensors, as CORA's withinTol
//
// See also: priv_within_tol.h

#include "cont_set/interval/private/priv_within_tol.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor priv_within_tol(const Tensor &a, const Tensor &b, double tol) {
    const Tensor difference = (a - b).abs();
    const Tensor bound = difference.full_like(difference.shape(), tol);
    const Tensor relative = difference.divide(a.abs().minimum(b.abs()));
    // equal values include the equal infinities, where the difference is NaN
    return difference.less_equal(bound).maximum(relative.less_equal(bound)).maximum(a.equal(b));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
