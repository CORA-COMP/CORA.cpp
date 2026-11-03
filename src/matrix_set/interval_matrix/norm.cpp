// norm - the norm of the largest matrix of an interval matrix, as CORA's intervalMatrix.norm
//
// Syntax:   Tensor n = A.norm(Tensor::Norm::fro);
// Inputs:   type - one, inf or fro; the 2-norm is NP-hard for interval matrices and refused
// Outputs:  n - norm(|center| + rad, type), (..., 1, 1): the greatest norm over the set
// See also: infinity_norm, abs
//
// CORA: matrixSet/@intervalMatrix/norm.m

#include "cora/matrix_set/interval_matrix/interval_matrix.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor IntervalMatrix::norm(Tensor::Norm type) const {
    detail::Entry entry;
    CORA_CHECK(type != Tensor::Norm::two, Error::wrong_value, [] {
        return std::string("norm: the 2-norm of an interval matrix is not supported, "
                           "use 1, Inf or 'fro'");
    });
    return int_.norm(type);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
