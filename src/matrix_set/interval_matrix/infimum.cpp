// infimum - the lower bound matrix of an interval matrix, as CORA's intervalMatrix.infimum
//
// Syntax:   Tensor M = A.infimum();
// Inputs:   -
// Outputs:  M - the entrywise smallest matrix of the set
// See also: center, abs
//
// CORA: matrixSet/@intervalMatrix/infimum.m

#include "cora/matrix_set/interval_matrix/interval_matrix.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor IntervalMatrix::infimum() const {
    detail::Entry entry;
    return int_.infimum();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
