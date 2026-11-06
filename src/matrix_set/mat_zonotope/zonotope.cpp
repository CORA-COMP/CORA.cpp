// zonotope - a matrix zonotope as a zonotope of stacked matrices, as CORA's matZonotope.zonotope
//
// Its points are the stacked matrices of A.
//
// Syntax:   Zonotope Z = A.zonotope();
// Inputs:   -
// Outputs:  Z - the zonotope of rows * cols dimensions: the center and generators of A, stacked
// See also: contains, representsa
//
// CORA: matrixSet/@matZonotope/zonotope.m
// CORA authors: Matthias Althoff, Tobias Ladner
// Authors:       Claude Sonnet 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/matrix_set/mat_zonotope/mat_zonotope.h"
#include "matrix_set/mat_zonotope/private/priv_stacking.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope MatZonotope::zonotope() const {
    detail::Entry entry;
    return Zonotope(priv_stack_columns(C_), priv_stack_generators(G_));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
