// zonotope - a matrix zonotope as a zonotope of stacked matrices, as CORA's matZonotope.zonotope
//
// Syntax:   Zonotope Z = A.zonotope();
// Inputs:   -
// Outputs:  Z - the zonotope of rows * cols dimensions whose center and generators are those of A,
//           each stacked column by column; its points are the stacked matrices of A
// See also: contains, representsa
//
// CORA: matrixSet/@matZonotope/zonotope.m

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
