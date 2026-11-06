// minus - the polytope shifted by a vector, as CORA's polytope.minus
//
// CORA defines minus for a vector subtrahend only; the Minkowski difference is
// minkowski_difference. The offsets become b - A v and be - Ae v, the vertices V - v.
//
// Syntax:   Pres = P.minus(v);   Pres = P - v;
// Inputs:   v - a column (..., n, 1)
// Outputs:  Pres - {x - v | x in P}
// See also: Polytope::plus, Polytope::minkowski_difference
//
// CORA: contSet/@polytope/minus.m
// CORA authors: Viktor Kotsev, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

// shifting by -v is adding the negated vector: the same numbers as subtracting
Polytope Polytope::minus(const Tensor &v) const {
    detail::Entry entry;
    return plus(-v);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
