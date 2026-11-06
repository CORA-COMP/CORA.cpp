// priv_expm_series - the Taylor series of the interval matrix exponential, from a given start
//
// The series continues from [A]^initial_order, its term divided by the factorial. The result also
// holds the powers first_order..N (at least the first) and the remainder.
//
// Syntax:   ExpmResult R = priv_expm_series(A, start, first_order, first_power, loop_from, N);
//           ExpmResult R = priv_expm_continued(A, N, initial_order, initial_power);
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
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/matrix_set/interval_matrix/interval_matrix.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

ExpmResult priv_expm_series(const IntervalMatrix &A, const IntervalMatrix &start,
                            int64_t first_order, const IntervalMatrix &first_power,
                            int64_t loop_from, int64_t max_order);

ExpmResult priv_expm_continued(const IntervalMatrix &A, int64_t max_order, int64_t initial_order,
                               const IntervalMatrix &initial_power);

/// [a] + [b], the sum of two interval matrices.
inline IntervalMatrix priv_sum(const IntervalMatrix &a, const IntervalMatrix &b) {
    return IntervalMatrix(a.interval() + b.interval());
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
