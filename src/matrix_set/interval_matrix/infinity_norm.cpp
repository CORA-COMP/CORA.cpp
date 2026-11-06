// infinity_norm - the infinity norm of the largest matrix of an interval matrix, as CORA's infNorm
//
// The greatest infinity norm over the matrices of the set.
//
// Syntax:   Tensor n = A.infinity_norm();
// Inputs:   -
// Outputs:  n - the largest absolute row sum of abs(A), (..., 1, 1)
// See also: abs, norm
//
// CORA: matrixSet/@intervalMatrix/infNorm.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/matrix_set/interval_matrix/interval_matrix.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor IntervalMatrix::infinity_norm() const {
    detail::Entry entry;
    return abs().matrix_norm(Tensor::Norm::inf);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
