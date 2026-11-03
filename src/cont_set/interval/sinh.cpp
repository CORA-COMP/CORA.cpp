// sinh - the hyperbolic sine of a box, as CORA's interval.sinh
//
// Syntax:   Interval R = I.sinh();
// Inputs:   -
// Outputs:  R - [sinh(inf), sinh(sup)] elementwise (sinh increases); nothing rounds outward
// See also: cosh, exp
//
// CORA: contSet/@interval/sinh.m

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Interval::sinh() const {
    detail::Entry entry;
    return {inf_.sinh(), sup_.sinh()};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
