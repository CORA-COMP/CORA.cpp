// expm - the enclosure of the matrix exponential over an interval matrix, as CORA's expm
//
// A is square. Given initial_order, r is not used, as in CORA. The enclosure is I + [A] + [A]^2 /
// 2 with the dependent terms of the second order, the terms up to max_order and the remainder
// box; the result also holds the powers [A]^i that were multiplied and the remainder (see
// mat_exp::ExpmResult).
//
// Syntax:   mat_exp::ExpmResult R = mat_exp::expm(A);   R = mat_exp::expm(A, max_order);   R =
// mat_exp::expm(A, r, max_order);
//           R = mat_exp::expm(A, r, max_order, initial_order, initial_power);
// Inputs:   max_order - the order of the series (default 10)
//           r - the step, nonzero (default 1), for the dependent terms of [A] / r
//           initial_order, initial_power - continue the series from [A]^initial_order
// Outputs:  R - enclosure: for every matrix M of A, exp(M) is inside
// See also: expm_independent, expm_mixed, exponential_remainder, dependent_terms
//
// CORA: matrixSet/@intervalMatrix/expm.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   17-November-2026
// Reviewed:      ---

#include "cont_dynamics/linear_sys/mat_exp/private/priv_expm_series.h"
#include "cora/cont_dynamics/linear_sys/mat_exp/mat_exp.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The series from order 2 with the dependent terms of [A] / r at the step r.
mat_exp::ExpmResult aux_dependent_series(const Interval &A, double r, int64_t max_order) {
    const auto [square, H] = mat_exp::dependent_terms(A * (1.0 / r), r);
    return priv_expm_series(A, H, 2, square, 3, max_order);
}

} // namespace


// ===========================================  MAIN  =========================================== //

mat_exp::ExpmResult mat_exp::expm(const Interval &A, int64_t max_order) {
    detail::Entry entry;
    if (entry.check()) mat_exp::check_square_order("expm", A, max_order);
    return aux_dependent_series(A, 1.0, max_order);
}

mat_exp::ExpmResult mat_exp::expm(const Interval &A, double r, int64_t max_order) {
    detail::Entry entry;
    if (entry.check()) mat_exp::check_square_order("expm", A, max_order);
    // r divides the matrix
    CORA_CHECK(r != 0, Error::wrong_value,
               [] { return std::string("expm: the step must be nonzero, it divides the matrix"); });
    return aux_dependent_series(A, r, max_order);
}

// the series continued from a power the caller has computed
mat_exp::ExpmResult mat_exp::expm(const Interval &A, double, int64_t max_order,
                                  int64_t initial_order, const Interval &initial_power) {
    detail::Entry entry;
    if (entry.check())
        mat_exp::check_continued_input("expm", A, max_order, initial_order, initial_power);
    return priv_expm_continued(A, max_order, initial_order, initial_power);
}

void mat_exp::check_square_order(const std::string &name, const Interval &A, int64_t max_order) {
    detail::Validation entry;
    CORA_CHECK(A.inf().size(-2) == A.inf().size(-1), Error::dimension_mismatch, [&] {
        return name + ": the matrix must be square, got the shape " + shape_text(A.inf().shape());
    });
    CORA_CHECK(max_order >= 0, Error::wrong_value, [&] {
        return name + ": the order must be nonnegative, got " + std::to_string(max_order);
    });
}

void mat_exp::check_continued_input(const std::string &name, const Interval &A, int64_t max_order,
                                    int64_t initial_order, const Interval &initial_power) {
    detail::Validation entry;
    // the matrix and the power it continues from are square and of one size
    CORA_CHECK(
        A.inf().size(-2) == A.inf().size(-1) && initial_power.inf().size(-2) == A.inf().size(-2) &&
            initial_power.inf().size(-1) == A.inf().size(-1),
        Error::dimension_mismatch, [&] {
            return name + ": the matrix and the initial power must be square of one size, got " +
                   shape_text(A.inf().shape()) + " and " + shape_text(initial_power.inf().shape());
        });
    CORA_CHECK(initial_order >= 0 && max_order >= 0, Error::wrong_value,
               [&] { return name + ": the orders must be nonnegative"; });
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
