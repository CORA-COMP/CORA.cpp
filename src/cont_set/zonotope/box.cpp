// box - the box around a zonotope, as a zonotope, as CORA's zonotope.box
//
// Syntax:   Zres = Z.box();
// Inputs:   -
// Outputs:  Zres - the center of Z with the n axis generators diag(sum_j |G_ij|): the interval hull
// See also: interval, abs
//
// CORA: contSet/@zonotope/box.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::box() const {
    detail::Entry entry;
    CORA_CHECK(elem_rank_ == 1, Error::not_supported,
               [] { return std::string("box: not for a matrix zonotope"); });
    // the radius, summed over the generators, as one axis generator per dimension
    return {c_, G_.abs().sum(-2).expand_dims(-1).diag()};
}


} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
