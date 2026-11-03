// ctranspose - the transpose of an interval matrix, as CORA's interval.ctranspose
//
// Syntax:   Interval R = I.ctranspose();
// Inputs:   -
// Outputs:  R - the bounds transposed (the bounds are real, so the conjugate is the identity);
//           nothing rounds
// See also: Tensor::matrix_transpose
//
// CORA: contSet/@interval/ctranspose.m

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::ctranspose() const {
    detail::Entry entry;
    return {inf_.matrix_transpose(), sup_.matrix_transpose()};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
