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
// Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cora/matrix_set/mat_zonotope/mat_zonotope.h"
#include "matrix_set/mat_zonotope/private/priv_stacking.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope MatZonotope::zonotope() const {
    detail::Entry entry;
    // the stacked generators, one per row
    const int64_t h = G_.size(-3);
    return {priv_stack_columns(C_).squeeze(-1), G_.matrix_transpose().reshape(priv_replace_tail(
                                                    G_, 3, {h, G_.size(-2) * G_.size(-1)}))};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
