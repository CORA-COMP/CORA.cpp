// expm - the enclosure of the matrix exponential over an interval matrix, as CORA's expm
//
// Syntax:   ExpmResult R = A.expm();   R = A.expm(max_order);   R = A.expm(r, max_order);
//           R = A.expm(r, max_order, initial_order, initial_power);
// Inputs:   max_order - the order of the series (default 10)
//           r - the step, nonzero (default 1),
//           for the dependent terms of [A] / r;  initial_order, initial_power - continue the series
//           from [A]^initial_order, as given (r is then not used, as in CORA);  A is square
// Outputs:  R - enclosure: for every matrix M of A, exp(M) is inside: I + [A] + [A]^2 / 2 with the
//           dependent terms of the second order, the terms up to max_order and the remainder box;
//           the powers [A]^i that were multiplied and the remainder (see ExpmResult)
// See also: expm_independent, expm_mixed, exponential_remainder, dependent_terms
//
// CORA: matrixSet/@intervalMatrix/expm.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/matrix_set/interval_matrix/interval_matrix.h"
#include "global/format.h"
#include "matrix_set/interval_matrix/private/priv_expm_series.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The series from order 2 with the dependent terms of [A] / r at the step r.
ExpmResult aux_dependent_series(const IntervalMatrix &A, double r, int64_t max_order) {
    const auto [square, H] = (A * (1.0 / r)).dependent_terms(r);
    return priv_expm_series(A, H, 2, square, 3, max_order);
}

} // namespace


// ===========================================  MAIN  =========================================== //

ExpmResult IntervalMatrix::expm(int64_t max_order) const {
    detail::Entry entry;
    CORA_CHECK(rows() == cols(), Error::dimension_mismatch, [&] {
        return "expm: the matrix must be square, got the shape " + shape_text(int_.inf().shape());
    });
    CORA_CHECK(max_order >= 0, Error::wrong_value, [&] {
        return "expm: the order must be nonnegative, got " + std::to_string(max_order);
    });
    return aux_dependent_series(*this, 1.0, max_order);
}

ExpmResult IntervalMatrix::expm(double r, int64_t max_order) const {
    detail::Entry entry;
    CORA_CHECK(rows() == cols(), Error::dimension_mismatch, [&] {
        return "expm: the matrix must be square, got the shape " + shape_text(int_.inf().shape());
    });
    // r divides the matrix
    CORA_CHECK(r != 0, Error::wrong_value,
               [] { return std::string("expm: the step must be nonzero, it divides the matrix"); });
    CORA_CHECK(max_order >= 0, Error::wrong_value, [&] {
        return "expm: the order must be nonnegative, got " + std::to_string(max_order);
    });
    return aux_dependent_series(*this, r, max_order);
}

// the series continued from a power the caller has computed
ExpmResult IntervalMatrix::expm(double, int64_t max_order, int64_t initial_order,
                                const IntervalMatrix &initial_power) const {
    detail::Entry entry;
    // the matrix and the power it continues from are square and of one size
    CORA_CHECK(rows() == cols() && initial_power.rows() == rows() && initial_power.cols() == cols(),
               Error::dimension_mismatch, [&] {
                   return "expm: the matrix and the initial power must be square of one size, "
                          "got " +
                          shape_text(int_.inf().shape()) + " and " +
                          shape_text(initial_power.interval().inf().shape());
               });
    CORA_CHECK(initial_order >= 0 && max_order >= 0, Error::wrong_value,
               [] { return std::string("expm: the orders must be nonnegative"); });
    return priv_expm_continued(*this, max_order, initial_order, initial_power);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
