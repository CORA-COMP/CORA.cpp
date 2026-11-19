// asinh - the inverse hyperbolic sine of a box, as CORA's interval.asinh
//
// Syntax:   Interval R = I.asinh();
// Inputs:   -
// Outputs:  R - [asinh(inf), asinh(sup)]
// See also: atan2, Tensor::asinh
//
// CORA: contSet/@interval/asinh.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::asinh() const {
    detail::Entry entry;
    return with(inf_.asinh(), sup_.asinh());
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
