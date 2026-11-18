// infinity_norm - the infinity norm of the largest matrix of an interval matrix, as CORA's infNorm
//
// The greatest infinity norm over the matrices of the set.
//
// Syntax:   Tensor n = A.infinity_norm();
// Inputs:   -
// Outputs:  n - the largest absolute row sum of abs(A), (..., 1, 1)
// See also: abs, norm, matmul
//
// CORA: matrixSet/@intervalMatrix/infNorm.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   17-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Interval::infinity_norm() const {
    detail::Entry entry;
    return abs().sup().matrix_norm(Tensor::Norm::inf);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
