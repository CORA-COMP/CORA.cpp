// dependent_terms - the square and the second-order Taylor terms of an interval matrix, as CORA's
//
// Syntax:   auto [sq, H] = A.dependent_terms(r);
// Inputs:   r - the step;  A is square
// Outputs:  sq - r^2 [A]^2, the exact square scaled by r^2;  H - I + [A] r + [A]^2 r^2 / 2, the
//           loop's H: entry (i, j) is A(i, j) (r + r^2 / 2 (A(i, i) + A(j, j))) + r^2 / 2 times
//           the terms that occur once, and on the diagonal the exact range of a r + a^2 r^2 / 2
//           (its minimum -1/2 where -1/r is in A(i, i)). Both are rebuilt from center and radius,
//           as CORA does (CORA adds the identity to the center before: the same up to rounding)
// See also: exact_square, expm
//
// CORA: matrixSet/@intervalMatrix/dependentTerms.m

#include "cora/matrix_set/interval_matrix/interval_matrix.h"
#include "cont_set/interval/private/priv_times.h"
#include "global/format.h"
#include "matrix_set/interval_matrix/private/priv_square_terms.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The value a r + (a r)^2 / 2 of the quadratic at a.
Tensor aux_quadratic(const Tensor &a, double r) {
    const Tensor m = a * r;
    return m + m.multiply(m) * 0.5;
}

/// The range of the quadratic over the diagonal entries [a, b] (..., n, 1): the least value of
/// the end points, -1/2 where the vertex -1/r may lie inside, and the greatest value.
std::pair<Tensor, Tensor> aux_quadratic_range(const Tensor &a, const Tensor &b, double r) {
    const double r_squared = r * r;
    const Tensor zero = a.zeros_like();
    const Tensor left_of_box = a * r_squared + r, right_of_box = b * r_squared + r;
    const Tensor off_vertex = left_of_box.greater(zero).maximum(right_of_box.less(zero));
    const Tensor qa = aux_quadratic(a, r), qb = aux_quadratic(b, r);
    return {Tensor::where(off_vertex, qa.minimum(qb), a.full_like(a.shape(), -0.5)),
            qa.maximum(qb)};
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::pair<IntervalMatrix, IntervalMatrix> IntervalMatrix::dependent_terms(double r) const {
    detail::Entry entry;
    CORA_CHECK(rows() == cols(), Error::dimension_mismatch, [&] {
        return "dependent_terms: the matrix must be square, got the shape " +
               shape_text(int_.inf().shape());
    });
    const Tensor &lo = int_.inf(), &hi = int_.sup();
    const Tensor scalar_r = lo.full_like({1, 1}, r);
    const Tensor half_r_squared = lo.full_like({1, 1}, 0.5 * (r * r));
    // r^2 times the exact square
    const Interval square = exact_square().interval();
    const Tensor r_squared = lo.full_like({1, 1}, r * r);
    const auto [sq_lo, sq_hi] = priv_times(r_squared, r_squared, square.inf(), square.sup());
    // the second-order term W: A(i, j) f(i, j) + r^2 / 2 times the terms that occur once
    const auto [pair_lo, pair_hi] = priv_diagonal_pair_sums(int_);
    const auto [f_lo, f_hi] = priv_times(half_r_squared, half_r_squared, pair_lo, pair_hi);
    const auto [off_lo, off_hi] = priv_times(lo, hi, scalar_r + f_lo, scalar_r + f_hi);
    const auto [range_lo, range_hi] = aux_quadratic_range(lo.diagonal(), hi.diagonal(), r);
    const Tensor identity = lo.eye_like(lo.size(-1));
    const auto [sum_lo, sum_hi] = priv_dependent_sums(int_);
    const auto [once_lo, once_hi] = priv_times(half_r_squared, half_r_squared, sum_lo, sum_hi);
    const Tensor w_lo = Tensor::where(identity, range_lo, off_lo) + once_lo;
    const Tensor w_hi = Tensor::where(identity, range_hi, off_hi) + once_hi;
    return {IntervalMatrix(Interval(sq_lo, sq_hi).interval_matrix()),
            IntervalMatrix(Interval(identity + w_lo, identity + w_hi).interval_matrix())};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
