// expm_independent - the matrix exponential with independent powers, as CORA's expmInd
//
// A is square. The sum starts from order 0 with the identity; every power is the interval matrix
// product. The result also holds the powers and the remainder (see ExpmResult).
//
// Syntax:   ExpmResult R = A.expm_independent(max_order);
//           R = A.expm_independent(max_order, initial_order, initial_power);
// Inputs:   max_order - the order of the series
//           initial_order, initial_power - continue from [A]^initial_order, as given
// Outputs:  R - enclosure: the sum of [A]^i / i! from the initial order to max_order, plus a box
// See also: expm, exponential_remainder
//
// CORA: matrixSet/@intervalMatrix/expmInd.m
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


// ===========================================  MAIN  =========================================== //

ExpmResult IntervalMatrix::expm_independent(int64_t max_order) const {
    detail::Entry entry;
    CORA_CHECK(rows() == cols(), Error::dimension_mismatch, [&] {
        return "expm_independent: the matrix must be square, got the shape " +
               shape_text(int_.inf().shape());
    });
    CORA_CHECK(max_order >= 0, Error::wrong_value, [&] {
        return "expm_independent: the order must be nonnegative, got " + std::to_string(max_order);
    });
    // CORA starts the powers at [A]^1 = A and the sum at the identity [A]^0
    return priv_expm_series(*this, matrix_power(0), 1, *this, 1, max_order);
}

ExpmResult IntervalMatrix::expm_independent(int64_t max_order, int64_t initial_order,
                                            const IntervalMatrix &initial_power) const {
    detail::Entry entry;
    CORA_CHECK(max_order >= 0 && initial_order >= 0, Error::wrong_value,
               [] { return std::string("expm_independent: the orders must be nonnegative"); });
    CORA_CHECK(rows() == cols() && initial_power.rows() == rows() && initial_power.cols() == cols(),
               Error::dimension_mismatch, [&] {
                   return "expm_independent: the matrix and the initial power must be square of "
                          "one size, got " +
                          shape_text(int_.inf().shape()) + " and " +
                          shape_text(initial_power.interval().inf().shape());
               });
    return priv_expm_continued(*this, max_order, initial_order, initial_power);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
