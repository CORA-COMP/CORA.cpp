// matmul - the product of an interval matrix with sets, matrices or numbers, as CORA's mtimes
//
// C is the bounds of the interval matrix product; for a number the bounds scaled (swapped if s <
// 0). In Z2, T is the center and S the radius of A.
//
// Syntax:   IntervalMatrix C = A.matmul(B);   C = A.matmul(M);   C = IntervalMatrix::matmul(M, A);
//           C = A.matmul(s);   Zonotope Z2 = A.matmul(Z);   EmptySet E2 = A.matmul(E);
// Inputs:   B - interval matrix
//           M - matrix of numbers
//           s - number
//           Z - zonotope;
//           E - empty set; the inner dimensions agree
// Outputs:  C - each entry the sum over k of the hull of the corner products
//           Z2 - the zonotope T c, [T G, diag(S sum|[c G]|)], an enclosure of [A] Z
//           E2 - the empty set of rows(A) dimensions
// See also: matrix_power, Zonotope::linear_map
//
// CORA: matrixSet/@intervalMatrix/mtimes.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/interval/private/priv_matmul.h"
#include "cora/matrix_set/interval_matrix/interval_matrix.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

IntervalMatrix IntervalMatrix::matmul(const IntervalMatrix &B) const {
    detail::Entry entry;
    CORA_CHECK(cols() == B.rows(), Error::dimension_mismatch, [&] {
        return "matmul: the inner dimensions differ, " + shape_text(int_.inf().shape()) +
               " times " + shape_text(B.int_.inf().shape());
    });
    return IntervalMatrix(priv_matmul(int_, B.int_));
}

// a matrix of numbers on the right or on the left, and a number
IntervalMatrix IntervalMatrix::matmul(const Tensor &M) const {
    detail::Entry entry;
    CORA_CHECK(M.ndim() >= 2 && M.size(-2) == cols(), Error::dimension_mismatch, [&] {
        return "matmul: the matrix needs " + std::to_string(cols()) + " rows, got the shape " +
               shape_text(M.shape());
    });
    auto [lo, hi] = Tensor::interval_matmul(int_.inf(), int_.sup(), M);
    return IntervalMatrix(Interval(std::move(lo), std::move(hi)));
}

IntervalMatrix IntervalMatrix::matmul(double s) const {
    detail::Entry entry;
    return IntervalMatrix(s * int_);
}

IntervalMatrix IntervalMatrix::matmul(const Tensor &M, const IntervalMatrix &A) {
    detail::Entry entry;
    CORA_CHECK(M.ndim() >= 2 && M.size(-1) == A.rows(), Error::dimension_mismatch, [&] {
        return "matmul: the matrix needs " + std::to_string(A.rows()) + " columns, got the shape " +
               shape_text(M.shape());
    });
    return IntervalMatrix(A.int_.linear_map(M));
}

// the sets that the product maps: a zonotope to a zonotope, the empty set to the empty set
Zonotope IntervalMatrix::matmul(const Zonotope &Z) const {
    detail::Entry entry;
    return Z.linear_map(int_);
}

EmptySet IntervalMatrix::matmul(const EmptySet &E) const {
    detail::Entry entry;
    return E.linear_map(int_);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
