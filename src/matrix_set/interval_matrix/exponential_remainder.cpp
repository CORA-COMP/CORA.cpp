// exponential_remainder - the remainder of the Taylor series of the matrix exponential, as CORA's
//
// It contains the remainder of every matrix of A; Inf everywhere where exp(|A|) is not a number.
//
// Syntax:   IntervalMatrix E = A.exponential_remainder(max_order);
// Inputs:   max_order - the order of the series that is summed;  A is square
// Outputs:  E - [-W, W] with W = exp(|A|) - sum_{i <= max_order} |A|^i / i!
// See also: expm, abs
//
// CORA: matrixSet/@intervalMatrix/exponentialRemainder.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/matrix_set/interval_matrix/interval_matrix.h"
#include "global/format.h"
#include "matrix_set/interval_matrix/private/priv_factorial.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

IntervalMatrix IntervalMatrix::exponential_remainder(int64_t max_order) const {
    detail::Entry entry;
    CORA_CHECK(rows() == cols(), Error::dimension_mismatch, [&] {
        return "exponential_remainder: the matrix must be square, got the shape " +
               shape_text(int_.inf().shape());
    });
    CORA_CHECK(max_order >= 0, Error::wrong_value, [&] {
        return "exponential_remainder: the order must be nonnegative, got " +
               std::to_string(max_order);
    });
    const Tensor M = abs();
    const Tensor exp_M = M.expm();
    Tensor power = M.eye_like(rows()), partial = power;
    for (int64_t i = 1; i <= max_order; ++i) {
        power = M.matmul(power);
        partial = partial + power / priv_factorial(i);
    }
    // a batch member whose exponential is not a number gets the unbounded remainder
    const Tensor width = (exp_M - partial).abs();
    const Tensor not_a_number = exp_M.isnan().max(-1, true).max(-2, true);
    const double infinity = std::numeric_limits<double>::infinity();
    const Tensor unbounded = width.full_like(width.shape(), infinity);
    const Tensor delta = Tensor::where(not_a_number, unbounded, width);
    return IntervalMatrix(M.zeros_like(), delta);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
