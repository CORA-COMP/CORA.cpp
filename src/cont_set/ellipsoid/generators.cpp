// generators - the generator matrix of an ellipsoid, as CORA's ellipsoid.generators
//
// Syntax:   Tensor G = E.generators();
// Inputs:   -
// Outputs:  G - U sqrt(D) for the singular value decomposition Q = U D U', so that G G' = Q
//           (..., n, n); the empty ellipsoid has the matrix without columns (n, 0)
// See also: ellipsoid_norm, Zonotope::generators
//
// CORA: contSet/@ellipsoid/generators.m

#include "cont_set/ellipsoid/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Ellipsoid::generators() const {
    detail::Entry entry;
    if (priv_is_empty(*this)) return q_.zeros_like({dim(), 0});
    const std::array<Tensor, 3> usv = Q_.svd();
    // U diag(sqrt(d)): each column of U scaled by the root of its singular value
    return usv[0].multiply(usv[1].sqrt().matrix_transpose());
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
