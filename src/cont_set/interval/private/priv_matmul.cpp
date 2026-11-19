// priv_matmul - the product of two interval matrices
//
// See also: priv_matmul.h
//
// CORA.cpp only: a helper of mtimes for interval matrices.
// Authors:       Tobias Ladner, Claude Sonnet 5.5, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/interval/private/priv_matmul.h"
#include "cont_set/interval/private/priv_join.h"
#include "cont_set/interval/private/priv_times.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The sum along axis -2 nested to the right, a(0) + (a(1) + (... + (a(n-1) + 0))), as CORALean's
/// sumDown and sumUp (a backend's own sum may add in another order, which the enclosure does not
/// contain; the seed 0 turns a sum of negative zeros into 0).
Tensor aux_nested_sum(const Tensor &t) {
    const int64_t n = t.size(-2);
    Tensor acc = t.slice(-2, n - 1, n);
    acc = acc + acc.zeros_like();
    for (int64_t k = n - 2; k >= 0; --k) acc = t.slice(-2, k, k + 1) + acc;
    return acc.squeeze(-2);
}

} // namespace


// ===========================================  MAIN  =========================================== //

Interval priv_matmul(const Interval &A, const Interval &B) {
    // A as (..., m, n, 1), B as (..., 1, n, p) (a box a column): all products (i, k, j)
    const Tensor a = priv_matrix_bounds(A, false), b = priv_matrix_bounds(A, true);
    const Tensor c = priv_matrix_bounds(B, false), d = priv_matrix_bounds(B, true);
    const auto [lo, hi] =
        priv_times(a.expand_dims(-1), b.expand_dims(-1), c.expand_dims(-3), d.expand_dims(-3));
    const Tensor low = aux_nested_sum(lo), high = aux_nested_sum(hi);
    // a box B gives a box
    if (B.elem_rank() == 1) return {low.squeeze(-1), high.squeeze(-1)};
    return Interval::matrix(low, high);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
