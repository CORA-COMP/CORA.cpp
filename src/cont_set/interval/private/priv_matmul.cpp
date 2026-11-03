// priv_matmul - the product of two interval matrices
//
// See also: priv_matmul.h

#include "cont_set/interval/private/priv_matmul.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The product with a NaN (0 * Inf) taken as 0.
Tensor aux_corner(const Tensor &a, const Tensor &b) {
    const Tensor product = a.multiply(b);
    return Tensor::where(product.isnan(), product.zeros_like(), product);
}

} // namespace


// ===========================================  MAIN  =========================================== //

Interval priv_matmul(const Interval &A, const Interval &B) {
    // A as (..., m, n, 1) and B as (..., 1, n, p): the products of all pairs (i, k, j)
    const Tensor lo = A.inf().expand_dims(-1), hi = A.sup().expand_dims(-1);
    const Tensor b_lo = B.inf().expand_dims(-3), b_hi = B.sup().expand_dims(-3);
    const Tensor c1 = aux_corner(lo, b_lo), c2 = aux_corner(lo, b_hi);
    const Tensor c3 = aux_corner(hi, b_lo), c4 = aux_corner(hi, b_hi);
    return {c1.minimum(c2).minimum(c3.minimum(c4)).sum(-2),
            c1.maximum(c2).maximum(c3.maximum(c4)).sum(-2)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
