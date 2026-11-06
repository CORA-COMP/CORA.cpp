// abs - the entrywise absolute value of center and generators, as CORA's zonotope.abs
//
// Syntax:   Zres = Z.abs();
// Inputs:   -
// Outputs:  Zres - the zonotope (|c|, |G|); not the image of Z under abs, but it holds the corner
//           |c| + sum_j |G_j|, which bounds |x| for every x in Z
// See also: box, interval
//
// CORA: contSet/@zonotope/abs.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::abs() const {
    detail::Entry entry;
    return {c_.abs(), G_.abs()};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
