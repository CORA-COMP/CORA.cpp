// matrix_power - the power of a square interval matrix, as CORA's interval.mpower
//
// Syntax:   Interval P = A.matrix_power(k);
// Inputs:   k - the exponent, a nonnegative integer;  A - a square interval matrix (..., n, n)
// Outputs:  P - the point matrix I for k = 0, else A (A (... A)) with k factors (the order of
//           CORALean's matPow; every product is the enclosure of priv_matmul). An interval of
//           1 x 1 is a scalar: Error::not_supported (CORA takes the elementwise power)
// See also: kron, linear_map
//
// CORA: contSet/@interval/mpower.m

#include "cont_set/interval/private/priv_matmul.h"
#include "cora/cont_set/interval/interval.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::matrix_power(int64_t exponent) const {
    detail::Entry entry;
    CORA_CHECK(inf_.size(-1) == inf_.size(-2), Error::wrong_value, [&] {
        return "matrix_power: the base must be a square matrix, got the shape " +
               shape_text(inf_.shape());
    });
    CORA_CHECK(exponent >= 0, Error::wrong_value, [&] {
        return "matrix_power: the exponent must be nonnegative, got " + std::to_string(exponent);
    });
    CORA_CHECK(inf_.size(-1) != 1, Error::not_supported, [] {
        return std::string("matrix_power: a 1 x 1 interval is a scalar, use its elementwise power");
    });
    if (exponent == 0) {
        const Tensor identity = inf_.eye_like(inf_.size(-1)) + inf_.zeros_like();
        return {identity, identity};
    }
    Interval P = *this;
    for (int64_t i = 1; i < exponent; ++i) P = priv_matmul(*this, P);
    return P;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
