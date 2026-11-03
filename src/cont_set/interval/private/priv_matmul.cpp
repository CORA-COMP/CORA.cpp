// priv_matmul - the product of two interval matrices
//
// See also: priv_matmul.h

#include "cont_set/interval/private/priv_matmul.h"
#include "cont_set/interval/private/priv_times.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval priv_matmul(const Interval &A, const Interval &B) {
    // A as (..., m, n, 1) and B as (..., 1, n, p): the products of all pairs (i, k, j)
    const auto [lo, hi] = priv_times(A.inf().expand_dims(-1), A.sup().expand_dims(-1),
                                     B.inf().expand_dims(-3), B.sup().expand_dims(-3));
    return {lo.sum(-2), hi.sum(-2)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
