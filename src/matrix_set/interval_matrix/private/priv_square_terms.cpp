// priv_square_terms - the sums that the square and the second-order terms share
//
// See also: priv_square_terms.h
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "matrix_set/interval_matrix/private/priv_square_terms.h"
#include "cont_set/interval/private/priv_times.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::pair<Tensor, Tensor> priv_dependent_sums(const Interval &A) {
    // the products of all (i, k, j), as in a matrix product, without the terms k = i and k = j
    const auto [lo, hi] = priv_times(A.inf().expand_dims(-1), A.sup().expand_dims(-1),
                                     A.inf().expand_dims(-3), A.sup().expand_dims(-3));
    const Tensor identity = A.inf().eye_like(A.inf().size(-1));
    const Tensor skipped = identity.expand_dims(-1).maximum(identity.expand_dims(0));
    const Tensor zero = lo.zeros_like();
    return {Tensor::where(skipped, zero, lo).sum(-2), Tensor::where(skipped, zero, hi).sum(-2)};
}

std::pair<Tensor, Tensor> priv_diagonal_pair_sums(const Interval &A) {
    const Tensor lo = A.inf().diagonal(), hi = A.sup().diagonal();
    return {lo + lo.matrix_transpose(), hi + hi.matrix_transpose()};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
