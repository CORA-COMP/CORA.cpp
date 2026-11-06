// box - the box around a zonotope, as a zonotope, as CORA's zonotope.box
//
// Syntax:   Zres = Z.box();
// Inputs:   -
// Outputs:  Zres - the center of Z with the n axis generators diag(sum_j |G_ij|): the interval hull
// See also: interval, abs
//
// CORA: contSet/@zonotope/box.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::box() const {
    detail::Entry entry;
    return {c_, G_.abs_sum().diag()};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
