// interval - the interval hull of a zonotope, as CORA's interval(Z)
//
// Syntax:   Interval I = Z.interval();
// Inputs:   -
// Outputs:  I - the smallest box containing Z: center c, radius sum_j |G_j|
// See also: Interval
//
// CORA: contSet/@zonotope/interval.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Zonotope::interval_impl() const {
    const Tensor r = G_.abs_sum(); // the radius: row sums of |G|
    return {c_ - r, c_ + r};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
