// mat_zonotope - a zonotope as a matrix zonotope of column matrices, as CORA's zonotope.matZonotope
//
// The center is the (n, 1) matrix, each generator the (n, 1) matrix of a column of G.
//
// Syntax:   MatZonotope M = Z.mat_zonotope();
// Inputs:   -
// Outputs:  M - the same set of n x 1 matrices
// See also: MatZonotope::zonotope
//
// CORA: contSet/@zonotope/matZonotope.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/matrix_set/mat_zonotope/mat_zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

MatZonotope Zonotope::mat_zonotope() const {
    detail::Entry entry;
    // the generators as (..., m, n, 1): one column matrix per generator
    const Tensor rows = G_.matrix_transpose();
    Shape shape = rows.shape();
    shape.push_back(1);
    return {c_, rows.reshape(shape)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
