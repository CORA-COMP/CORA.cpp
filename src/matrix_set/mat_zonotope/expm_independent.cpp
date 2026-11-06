// expm_independent - the matrix exponential of a matrix zonotope with independent powers
//
// The sum of A^i / i! for i = 0 to max_order (each power the product of the previous with A) plus
// the remainder box of the interval matrix around A.
//
// Syntax:   MatZonotope E = A.expm_independent(max_order);
// Inputs:   max_order - the order of the series, nonnegative;  A is square
// Outputs:  E - an enclosure of exp(M) for every matrix M of A, as a matrix zonotope
// See also: expm, expm_mixed, IntervalMatrix::exponential_remainder
//
// CORA: matrixSet/@matZonotope/expmInd.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/matrix_set/mat_zonotope/mat_zonotope.h"
#include "global/format.h"
#include "matrix_set/interval_matrix/private/priv_factorial.h"
#include "matrix_set/mat_zonotope/private/priv_from_interval_matrix.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

MatZonotope MatZonotope::expm_independent(int64_t max_order) const {
    detail::Entry entry;
    CORA_CHECK(rows() == cols(), Error::dimension_mismatch, [&] {
        return "expm_independent: the matrix zonotope must be square, got the shape " +
               shape_text(C_.shape());
    });
    CORA_CHECK(max_order >= 0, Error::wrong_value, [&] {
        return "expm_independent: the order must be nonnegative, got " + std::to_string(max_order);
    });
    MatZonotope sum = matrix_power(0), power = matrix_power(0);
    for (int64_t i = 1; i <= max_order; ++i) {
        power = power * (*this);
        sum = sum + power * (1.0 / priv_factorial(i));
    }
    const IntervalMatrix remainder = interval_matrix().exponential_remainder(max_order);
    return sum + priv_from_interval_matrix(remainder);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
