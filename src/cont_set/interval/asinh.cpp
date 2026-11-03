// asinh - the inverse hyperbolic sine of a box, as CORA's interval.asinh
//
// Syntax:   Interval R = I.asinh();
// Inputs:   -
// Outputs:  R - [asinh(inf), asinh(sup)]
// See also: atan2, Tensor::asinh
//
// CORA: contSet/@interval/asinh.m

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::asinh() const {
    detail::Entry entry;
    return {inf_.asinh(), sup_.asinh()};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
