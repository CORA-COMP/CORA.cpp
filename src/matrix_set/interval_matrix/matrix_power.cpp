// matrix_power - the power of a square interval matrix, as CORA's intervalMatrix.mpower
//
// Every product is the interval matrix product (CORALean's matPow; CORA multiplies from the left,
// the other association of the same factors).
//
// Syntax:   IntervalMatrix P = A.matrix_power(k);
// Inputs:   k - the exponent, a nonnegative integer;  A is square
// Outputs:  P - the point matrix I for k = 0, A for k = 1, else A (A (... A)) with k factors
// See also: matmul, Interval::matrix_power
//
// CORA: matrixSet/@intervalMatrix/mpower.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/matrix_set/interval_matrix/interval_matrix.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

IntervalMatrix IntervalMatrix::matrix_power(int64_t exponent) const {
    detail::Entry entry;
    CORA_CHECK(rows() == cols(), Error::dimension_mismatch, [&] {
        return "matrix_power: the matrix must be square, got the shape " +
               shape_text(int_.inf().shape());
    });
    CORA_CHECK(exponent >= 0, Error::wrong_value, [&] {
        return "matrix_power: the exponent must be nonnegative, got " + std::to_string(exponent);
    });
    if (exponent == 0) return IntervalMatrix(int_.inf().eye_like(rows()) + int_.inf().zeros_like());
    IntervalMatrix P = *this;
    for (int64_t i = 1; i < exponent; ++i) P = matmul(P);
    return P;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
