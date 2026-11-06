// transpose - the transpose of a matrix zonotope, as CORA's matZonotope.transpose
//
// Syntax:   MatZonotope T = A.transpose();
// Inputs:   -
// Outputs:  T - the center and every generator transposed: the set of the transposed matrices
// See also: matmul
//
// CORA: matrixSet/@matZonotope/transpose.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/matrix_set/mat_zonotope/mat_zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

MatZonotope MatZonotope::transpose() const {
    detail::Entry entry;
    return MatZonotope(C_.matrix_transpose(), G_.matrix_transpose());
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
