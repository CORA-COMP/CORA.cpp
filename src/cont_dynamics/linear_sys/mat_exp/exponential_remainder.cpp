// exponential_remainder - the remainder of the Taylor series of the matrix exponential, as CORA's
//
// It contains the remainder of every matrix of A; Inf everywhere where exp(|A|) is not a number.
//
// Syntax:   Interval E = mat_exp::exponential_remainder(A, max_order);
// Inputs:   max_order - the order of the series that is summed;  A is square
// Outputs:  E - [-W, W] with W = exp(|A|) - sum_{i <= max_order} |A|^i / i!
// See also: expm, abs
//
// CORA: matrixSet/@intervalMatrix/exponentialRemainder.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   17-November-2026
// Reviewed:      ---

#include "cont_dynamics/linear_sys/mat_exp/private/priv_factorial.h"
#include "cora/cont_dynamics/linear_sys/mat_exp/mat_exp.h"
#include "global/format.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval mat_exp::exponential_remainder(const Interval &A, int64_t max_order) {
    detail::Entry entry;
    CORA_CHECK(A.inf().size(-2) == A.inf().size(-1), Error::dimension_mismatch, [&] {
        return "exponential_remainder: the matrix must be square, got the shape " +
               shape_text(A.inf().shape());
    });
    CORA_CHECK(max_order >= 0, Error::wrong_value, [&] {
        return "exponential_remainder: the order must be nonnegative, got " +
               std::to_string(max_order);
    });
    const Tensor M = A.abs().sup();
    const Tensor exp_M = M.expm();
    Tensor power = M.eye_like(A.inf().size(-2)), partial = power;
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
    return Interval::from_center_delta(M.zeros_like(), delta);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
