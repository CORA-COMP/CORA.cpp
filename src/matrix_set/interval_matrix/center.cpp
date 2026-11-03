// center - the center matrix of an interval matrix, as CORA's intervalMatrix.center
//
// Syntax:   Tensor C = A.center();
// Inputs:   -
// Outputs:  C - (inf + sup) / 2, a matrix of the set
// See also: infimum, abs
//
// CORA: matrixSet/@intervalMatrix/center.m

#include "cora/matrix_set/interval_matrix/interval_matrix.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor IntervalMatrix::center() const {
    detail::Entry entry;
    return int_.center();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
