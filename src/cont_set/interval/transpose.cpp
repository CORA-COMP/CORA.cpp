// transpose - the transpose of an interval matrix, as CORA's interval.transpose
//
// Syntax:   Interval R = I.transpose();
// Inputs:   -
// Outputs:  R - the bounds transposed; nothing rounds
// See also: ctranspose, Tensor::matrix_transpose
//
// CORA: contSet/@interval/transpose.m

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::transpose() const {
    detail::Entry entry;
    return {inf_.matrix_transpose(), sup_.matrix_transpose()};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
