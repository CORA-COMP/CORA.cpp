// matmul - the product of an interval matrix with an interval matrix or a matrix, as CORA's mtimes
//
// An interval matrix is an Interval of matrix bounds. The other products of CORA's mtimes are
// elsewhere: M [A] is linear_map, s [A] is s * A, [A] Z is Zonotope::linear_map(A) and [A] times
// the empty set is EmptySet::linear_map(A).
//
// Syntax:   Interval C = A.matmul(B);   C = A.matmul(M);   C = A * B;   C = A * M;
// Inputs:   B - interval matrix, its rows the columns of A
//           M - matrix of numbers, its rows the columns of A
// Outputs:  C - each entry the sum over k of the hull of the corner products
// See also: linear_map, matrix_power, Zonotope::linear_map
//
// CORA: matrixSet/@intervalMatrix/mtimes.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   17-November-2026
// Reviewed:      ---

#include "cont_set/interval/private/priv_matmul.h"
#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::matmul(const Interval &B) const {
    detail::Entry entry;
    CORA_CHECK(inf_.ndim() >= 2 && B.inf_.ndim() >= 2 && inf_.size(-1) == B.inf_.size(-2),
               Error::dimension_mismatch, [&] {
                   return "matmul: the inner dimensions differ, " + shape_text(inf_.shape()) +
                          " times " + shape_text(B.inf_.shape());
               });
    return priv_matmul(*this, B);
}

// a matrix of numbers on the right; on the left it is linear_map
Interval Interval::matmul(const Tensor &M) const {
    detail::Entry entry;
    CORA_CHECK(M.ndim() >= 2 && M.size(-2) == inf_.size(-1), Error::dimension_mismatch, [&] {
        return "matmul: the matrix needs " + std::to_string(inf_.size(-1)) +
               " rows, got the shape " + shape_text(M.shape());
    });
    auto [lo, hi] = Tensor::interval_matmul(inf_, sup_, M);
    return {std::move(lo), std::move(hi)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
