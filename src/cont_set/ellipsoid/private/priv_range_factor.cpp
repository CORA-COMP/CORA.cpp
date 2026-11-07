// priv_range_factor - the factor of an ellipsoid on its range, as CORA's U_r sqrt(D_r)
//
// The first r columns of generators(), r the rank of Q: a matrix (n, r) with G G' = Q. Not a batch.
//
// Syntax:   Tensor G = priv_range_factor(E);
// Inputs:   E - an unbatched, nonempty ellipsoid
// Outputs:  G - (n, r), without columns for Q = 0
// See also: priv_rank, Ellipsoid::generators
//
// CORA: contSet/@ellipsoid/zonotope.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor priv_range_factor(const Ellipsoid &E) {
    const int64_t rank = static_cast<int64_t>(priv_rank(E.Q(), E.tol()).item());
    return E.generators().slice(1, 0, rank);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
