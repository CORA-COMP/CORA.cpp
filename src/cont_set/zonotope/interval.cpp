// interval - the interval hull of a zonotope, as CORA's interval(Z)
//
// Syntax:   Interval I = Z.interval();
// Inputs:   -
// Outputs:  I - the smallest box containing Z: center c, radius sum_j |G_j|
// See also: Interval
//
// CORA: contSet/@zonotope/interval.m
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
    const Tensor r = G_.abs().sum(-1 - int(elem_rank_)); // the radius: the sum of |G| over h
    // layout: #163 step 2 - the box of a vector zonotope has column bounds (..., n, 1)
    if (elem_rank_ == 1) return {(c_ - r).expand_dims(-1), (c_ + r).expand_dims(-1)};
    return {c_ - r, c_ + r};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
