// generators - the generator matrix of an ellipsoid, as CORA's ellipsoid.generators
//
// One generator per row, as a zonotope's (#163): CORA's matrix transposed. The empty ellipsoid
// has none, (0, n).
//
// Syntax:   Tensor G = E.generators();
// Inputs:   -
// Outputs:  G - sqrt(D) U' for the singular value decomposition Q = U D U', G' G = Q (..., n, n)
// See also: ellipsoid_norm, Zonotope::generators
//
// CORA: contSet/@ellipsoid/generators.m
// CORA authors: Adrian Kulmburg
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Ellipsoid::generators() const {
    detail::Entry entry;
    if (priv_is_empty(*this)) return q_.zeros_like({0, dim()});
    return priv_generator_columns(Q_).matrix_transpose();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
