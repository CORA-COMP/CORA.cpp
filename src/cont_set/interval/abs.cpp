// abs - the elementwise absolute value of a box, as CORA's interval.abs
//
// Syntax:   Interval A = I.abs();
// Inputs:   -
// Outputs:  A - the image {|x|}: [max(0, inf, -sup), max(-inf, sup)]; nothing rounds
// See also: Zonotope::interval
//
// CORA: contSet/@interval/abs.m
// CORA: matrixSet/@intervalMatrix/abs.m
// CORA authors: Matthias Althoff, Mark Wetzlinger
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::abs() const {
    detail::Entry entry;
    return with(inf_.maximum(-sup_).maximum(0.0), (-inf_).maximum(sup_));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
