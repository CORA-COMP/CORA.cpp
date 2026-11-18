// matrix_power - the power of a square matrix zonotope, as CORA's matZonotope.mpower
//
// Every product is matmul: an enclosure of the k-th power of every matrix of A.
//
// Syntax:   Zonotope P = mat_exp::matrix_power(A, k);
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

#include "cont_set/zonotope/private/priv_stacking.h"
#include "cora/cont_dynamics/linear_sys/mat_exp/mat_exp.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope mat_exp::matrix_power(const Zonotope &A, int64_t exponent) {
    detail::Entry entry;
    CORA_CHECK(A.center().size(-2) == A.center().size(-1), Error::dimension_mismatch, [&] {
        return "matrix_power: the matrix zonotope must be square, got the shape " +
               shape_text(A.center().shape());
    });
    CORA_CHECK(exponent >= 0, Error::wrong_value, [&] {
        return "matrix_power: the exponent must be nonnegative, got " + std::to_string(exponent);
    });
    if (exponent == 0)
        return Zonotope::matrix(A.center().eye_like(A.center().size(-2)) + A.center().zeros_like(),
                                priv_no_generators(A.center()));
    Zonotope P = A;
    for (int64_t i = 2; i <= exponent; ++i) P = P.matmul(A);
    return P;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
