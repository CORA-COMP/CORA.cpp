// rad - the radius of each coordinate of a box, as CORA's interval.rad
//
// Syntax:   Tensor r = I.rad();
// Inputs:   -
// Outputs:  r - (sup - inf) / 2 elementwise, of the shape of the bounds
// See also: center, radius
//
// CORA: contSet/@interval/rad.m

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Interval::rad() const {
    detail::Entry entry;
    return (sup_ - inf_) * 0.5;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
