// tanh - the hyperbolic tangent of a box, as CORA's interval.tanh
//
// Syntax:   Interval R = I.tanh();
// Inputs:   -
// Outputs:  R - [tanh(inf), tanh(sup)] elementwise (tanh increases); nothing rounds outward
// See also: sinh, cosh
//
// CORA: contSet/@interval/tanh.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::tanh() const {
    detail::Entry entry;
    return {inf_.tanh(), sup_.tanh()};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
