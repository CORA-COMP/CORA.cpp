// transpose - the transpose of a matrix zonotope, as CORA's matZonotope.transpose
//
// Syntax:   Zonotope T = A.transpose();
// Inputs:   -
// Outputs:  T - the center and every generator transposed: the set of the transposed matrices
// See also: matmul, Zonotope::matrix
//
// CORA: matrixSet/@matZonotope/transpose.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::transpose() const {
    detail::Entry entry;
    CORA_CHECK(elem_rank_ == 2, Error::not_supported,
               [] { return std::string("transpose: a matrix zonotope (Zonotope::matrix)"); });
    return matrix(c_.matrix_transpose(), G_.matrix_transpose());
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
