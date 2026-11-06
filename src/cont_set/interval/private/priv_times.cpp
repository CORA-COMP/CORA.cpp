// priv_times - the entrywise product of two boxes
//
// See also: priv_times.h
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/interval/private/priv_times.h"

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

std::pair<Tensor, Tensor> priv_times(const Tensor &a_inf, const Tensor &a_sup, const Tensor &b_inf,
                                     const Tensor &b_sup) {
    const Tensor c1 = aux_corner(a_inf, b_inf), c2 = aux_corner(a_inf, b_sup);
    const Tensor c3 = aux_corner(a_sup, b_inf), c4 = aux_corner(a_sup, b_sup);
    return {c1.minimum(c2).minimum(c3.minimum(c4)), c1.maximum(c2).maximum(c3.maximum(c4))};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
