// abs - the entrywise absolute value of center and generators, as CORA's zonotope.abs
//
// The result is not the image of Z under abs, but it bounds |x| for every x in Z.
//
// Syntax:   Zres = Z.abs();
// Inputs:   -
// Outputs:  Zres - the zonotope (|c|, |G|), which holds the corner |c| + sum_j |G_j|
// See also: box, interval
//
// CORA: contSet/@zonotope/abs.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::abs() const {
    detail::Entry entry;
    return {c_.abs(), G_.abs(), elem_rank_};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
