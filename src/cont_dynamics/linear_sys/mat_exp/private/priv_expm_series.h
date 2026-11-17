// priv_expm_series - the Taylor series of the interval matrix exponential, from a given start
//
// The series continues from [A]^initial_order, its term divided by the factorial. The result also
// holds the powers first_order..N (at least the first) and the remainder.
//
// Syntax:   auto R = priv_expm_series(A, start, first_order, first_power, loop_from, N);
//           auto R = priv_expm_continued(A, N, initial_order, initial_power);
//           Interval P = priv_power(A, k);
//           mat_exp::check_series_input(A, N, style, p, allow_expm);
// Inputs:   A - square interval matrix
//           start - the sum of the terms already counted
//           first_order, first_power - the power of A that the list of powers starts with
//           loop_from - the first order whose term is added
//           N - the last order added
// Outputs:  R - start plus the terms loop_from..N, [A]^i / i!, plus the remainder box
// See also: expm, expm_independent, expm_mixed
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   17-November-2026
// Reviewed:      ---

#pragma once

#include "cora/cont_dynamics/linear_sys/mat_exp/mat_exp.h"

#include <string>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

mat_exp::ExpmResult priv_expm_series(const Interval &A, const Interval &start, int64_t first_order,
                                     const Interval &first_power, int64_t loop_from,
                                     int64_t max_order);

mat_exp::ExpmResult priv_expm_continued(const Interval &A, int64_t max_order, int64_t initial_order,
                                        const Interval &initial_power);

/// [A]^k for k >= 1 by k - 1 interval matrix products from the left (1 x 1 included).
Interval priv_power(const Interval &A, int64_t exponent);

namespace mat_exp {
/// The checks of the two series entries of CORA's expm algorithms (style 4 only if allow_expm);
/// defined beside expm_taylor_horner_scaling, since private code never checks.
void check_series_input(const Interval &A, int64_t max_order, int64_t style,
                        int64_t scaling_exponent, bool allow_expm);
/// The checks of a series from the start: a square matrix and a nonnegative order.
void check_square_order(const std::string &name, const Interval &A, int64_t max_order);
/// The checks of a series continued from a computed power: square, one size, nonnegative orders.
void check_continued_input(const std::string &name, const Interval &A, int64_t max_order,
                           int64_t initial_order, const Interval &initial_power);
} // namespace mat_exp

/// [a] + [b], the sum of two interval matrices.
inline Interval priv_sum(const Interval &a, const Interval &b) { return a + b; }

/// The identity matrix [I, I] of the shape of the square A.
inline Interval priv_identity(const Interval &A) {
    const Tensor I = A.inf().eye_like(A.inf().size(-1)) + A.inf().zeros_like();
    return {I, I};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
