// norm - an upper bound of the norm of a matrix zonotope, as CORA's matZonotope.norm
//
// Syntax:   Tensor n = A.norm(Tensor::Norm::fro);
// Inputs:   type - one, inf or fro; the 2-norm is NP-hard for interval matrices and refused
// Outputs:  n - the norm of the interval matrix around A (norm(|center| + rad, type)), (..., 1, 1):
//           no matrix of A has a larger norm, but the bound is not attained in general
// See also: interval_matrix, IntervalMatrix::norm
//
// CORA: matrixSet/@matZonotope/norm.m
// CORA authors: Matthias Althoff, Victor Gassmann
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/matrix_set/mat_zonotope/mat_zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor MatZonotope::norm(Tensor::Norm type) const {
    detail::Entry entry;
    CORA_CHECK(type != Tensor::Norm::two, Error::wrong_value, [] {
        return std::string("norm: the 2-norm of an interval matrix is not supported, "
                           "use 1, Inf or 'fro'");
    });
    return interval_matrix().norm(type);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
