// matrix_power - the power of a square interval matrix, as CORA's interval.mpower
//
// The order is that of CORALean's matPow; every product is the enclosure of priv_matmul. A 1 x 1
// interval is a matrix too: [-1, 2]^2 = [-2, 4], sound but looser than the exact square [0, 4]
// that the elementwise power gives (CORA's interval takes that one for a scalar).
//
// Syntax:   Interval P = A.matrix_power(k);
// Inputs:   k - the exponent, a nonnegative integer
//           A - a square interval matrix (..., n, n)
// Outputs:  P - the point matrix I for k = 0, else A (A (... A)) with k factors
// See also: kron, linear_map
//
// CORA: contSet/@interval/mpower.m
// CORA: matrixSet/@intervalMatrix/mpower.m
// CORA authors: Matthias Althoff, Dmitry Grebenyuk
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/interval/private/priv_matmul.h"
#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::matrix_power(int64_t exponent) const {
    detail::Entry entry;
    CORA_CHECK(dim(1) == dim(2), Error::dimension_mismatch, [&] {
        return "matrix_power: the base must be a square matrix, got the shape " +
               shape_text(inf_.shape());
    });
    CORA_CHECK(exponent >= 0, Error::wrong_value, [&] {
        return "matrix_power: the exponent must be nonnegative, got " + std::to_string(exponent);
    });
    if (exponent == 0) {
        const Tensor identity = inf_.eye_like(dim(2)) + inf_matrix().zeros_like();
        return from_matrix(identity, identity);
    }
    Interval P = *this;
    for (int64_t i = 1; i < exponent; ++i) P = priv_matmul(*this, P);
    return P;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
