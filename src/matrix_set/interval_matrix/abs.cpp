// abs - the entrywise largest absolute value of an interval matrix, as CORA's intervalMatrix.abs
//
// Syntax:   Tensor M = A.abs();
// Inputs:   -
// Outputs:  M - the matrix of max(|inf|, |sup|): the greatest |M(i, j)| over the matrices of A
// See also: center, infinity_norm
//
// CORA: matrixSet/@intervalMatrix/abs.m

#include "cora/matrix_set/interval_matrix/interval_matrix.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor IntervalMatrix::abs() const {
    detail::Entry entry;
    return int_.abs().sup();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
