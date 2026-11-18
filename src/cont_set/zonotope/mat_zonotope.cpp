// mat_zonotope - a zonotope as a matrix zonotope of column matrices, as CORA's zonotope.matZonotope
//
// The center is the (n, 1) matrix, each generator the (n, 1) matrix of a column of G.
//
// Syntax:   Zonotope M = Z.mat_zonotope();
// Inputs:   -
// Outputs:  M - the same set of n x 1 matrices
// See also: stacked, Zonotope::matrix
//
// CORA: contSet/@zonotope/matZonotope.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       06-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::mat_zonotope() const {
    detail::Entry entry;
    CORA_CHECK(elem_rank_ == 1, Error::not_supported,
               [] { return std::string("mat_zonotope: a vector zonotope"); });
    // the generators as (..., m, n, 1): one column matrix per generator
    Shape shape = G_.shape();
    shape.push_back(1);
    return matrix(c_.expand_dims(-1), G_.reshape(shape));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
