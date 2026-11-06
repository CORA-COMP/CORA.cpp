// matrix_power - the power of a square matrix zonotope, as CORA's matZonotope.mpower
//
// Every product is matmul: an enclosure of the k-th power of every matrix of A.
//
// Syntax:   MatZonotope P = A.matrix_power(k);
// Inputs:   k - the exponent, a nonnegative integer;  A is square
// Outputs:  P - the identity for k = 0, A for k = 1, else ((A A) A) ... with k factors
// See also: matmul, powers
//
// CORA: matrixSet/@matZonotope/mpower.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/matrix_set/mat_zonotope/mat_zonotope.h"
#include "global/format.h"
#include "matrix_set/mat_zonotope/private/priv_stacking.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

MatZonotope MatZonotope::matrix_power(int64_t exponent) const {
    detail::Entry entry;
    CORA_CHECK(rows() == cols(), Error::dimension_mismatch, [&] {
        return "matrix_power: the matrix zonotope must be square, got the shape " +
               shape_text(C_.shape());
    });
    CORA_CHECK(exponent >= 0, Error::wrong_value, [&] {
        return "matrix_power: the exponent must be nonnegative, got " + std::to_string(exponent);
    });
    if (exponent == 0)
        return MatZonotope(C_.eye_like(rows()) + C_.zeros_like(), priv_no_generators(C_));
    MatZonotope P = *this;
    for (int64_t i = 2; i <= exponent; ++i) P = P.matmul(*this);
    return P;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
