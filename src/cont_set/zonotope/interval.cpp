// interval - the interval hull of a zonotope, as CORA's interval(Z)
//
// Syntax:   Interval I = Z.interval();
// Inputs:   -
// Outputs:  I - the smallest box containing Z: center c, radius sum_j |G_j|
// See also: Interval
//
// CORA: contSet/@zonotope/interval.m
// CORA: matrixSet/@matZonotope/intervalMatrix.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Zonotope::interval_impl() const {
    // the box of a vector zonotope has column bounds (..., n, 1): its radius is one fused sum
    if (elem_rank_ == 1) {
        const Tensor c = c_.expand_dims(-1), r = G_.abs_sum_rows();
        return {c - r, c + r};
    }
    const Tensor r = G_.abs().sum(-3); // the radius: the sum of |G| over h
    return {c_ - r, c_ + r};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
