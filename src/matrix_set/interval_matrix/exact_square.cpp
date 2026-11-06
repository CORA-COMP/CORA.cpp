// exact_square - the entrywise range of M * M over an interval matrix, as CORA's exactSquare
//
// Each entry of M occurs once in it but the diagonal, which is squared as a number, so the
// diagonal term is exact; the lower end of a diagonal square is 0 unless the interval is on one
// side of 0.
//
// Syntax:   IntervalMatrix S = A.exact_square();
// Inputs:   -   (A is square)
// Outputs:  S - entry (i, j) is the hull of the values of (M * M)(i, j) over the matrices M of A
// See also: dependent_terms, matmul
//
// CORA: matrixSet/@intervalMatrix/exactSquare.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/interval/private/priv_times.h"
#include "cora/matrix_set/interval_matrix/interval_matrix.h"
#include "global/format.h"
#include "matrix_set/interval_matrix/private/priv_square_terms.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

IntervalMatrix IntervalMatrix::exact_square() const {
    detail::Entry entry;
    CORA_CHECK(rows() == cols(), Error::dimension_mismatch, [&] {
        return "exact_square: the matrix must be square, got the shape " +
               shape_text(int_.inf().shape());
    });
    const Tensor &lo = int_.inf(), &hi = int_.sup();
    const auto [pair_lo, pair_hi] = priv_diagonal_pair_sums(int_);
    const auto [sum_lo, sum_hi] = priv_dependent_sums(int_);
    // off the diagonal: A(i, j) (A(i, i) + A(j, j)) plus the terms that occur once
    const auto [off_lo, off_hi] = priv_times(lo, hi, pair_lo, pair_hi);
    // on it: the square of the diagonal entry, 0 at the lower end if the entry straddles 0
    const Tensor a = lo.diagonal(), b = hi.diagonal(), zero = a.zeros_like();
    const Tensor aa = a.multiply(a), bb = b.multiply(b);
    const Tensor one_sided = a.greater_equal(zero).maximum(b.less_equal(zero));
    const Tensor diagonal_lo = Tensor::where(one_sided, aa.minimum(bb), zero);
    const Tensor identity = lo.eye_like(lo.size(-1));
    const Tensor head_lo = Tensor::where(identity, diagonal_lo, off_lo);
    const Tensor head_hi = Tensor::where(identity, aa.maximum(bb), off_hi);
    return IntervalMatrix(Interval(head_lo + sum_lo, head_hi + sum_hi));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
