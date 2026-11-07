// priv_matmul - the product of two interval matrices
//
// See also: priv_matmul.h
//
// Authors:       Tobias Ladner, Claude Sonnet 5.5, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/interval/private/priv_matmul.h"
#include "cont_set/interval/private/priv_times.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The sum along axis -2 nested to the right, a(0) + (a(1) + (...)), as CORALean's sumDown and
/// sumUp (a backend's own sum may add in another order, which the enclosure does not contain).
Tensor aux_nested_sum(const Tensor &t) {
    const int64_t n = t.size(-2);
    Tensor acc = t.slice(-2, n - 1, n);
    for (int64_t k = n - 2; k >= 0; --k) acc = t.slice(-2, k, k + 1) + acc;
    return acc.squeeze(-2);
}

} // namespace


// ===========================================  MAIN  =========================================== //

Interval priv_matmul(const Interval &A, const Interval &B) {
    // A as (..., m, n, 1) and B as (..., 1, n, p): the products of all pairs (i, k, j)
    const auto [lo, hi] = priv_times(A.inf().expand_dims(-1), A.sup().expand_dims(-1),
                                     B.inf().expand_dims(-3), B.sup().expand_dims(-3));
    return {aux_nested_sum(lo), aux_nested_sum(hi)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
