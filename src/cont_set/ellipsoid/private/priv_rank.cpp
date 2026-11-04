// priv_rank - the rank of symmetric matrices by their singular values, as CORA's rank(E)
//
// Syntax:   Tensor r = priv_rank(Q, tol);
// Inputs:   Q - symmetric matrices (..., n, n);  tol - singular values below tol times the
//           largest count as zero
// Outputs:  r - the number of singular values above zero and at least tol times the largest,
//           as (..., 1, 1)
// See also: priv.h, Ellipsoid::is_full_dim
//
// CORA: contSet/@ellipsoid/rank.m

#include "cont_set/ellipsoid/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor priv_rank(const Tensor &Q, double tol) {
    const Tensor d = Q.svd()[1];
    const Tensor threshold = d.slice(-2, 0, 1) * tol;
    const Tensor counted = d.greater(d.zeros_like()).multiply(d.greater_equal(threshold));
    return counted.sum(-2, true);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
