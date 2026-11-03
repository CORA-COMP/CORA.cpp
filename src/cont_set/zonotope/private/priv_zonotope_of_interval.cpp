// priv_zonotope_of_interval - the zonotope of a box, as CORA's interval.zonotope
//
// Syntax:   Zonotope Z = priv_zonotope_of_interval(I);
// Inputs:   I - a box (..., n, 1)
// Outputs:  Z - center of I, one axis generator of radius rad(I) per dimension with a radius > 0
// See also: Zonotope::plus
//
// CORA.cpp only: the conversion is CORA's interval.zonotope, which the interval batch ports

#include "cont_set/zonotope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope priv_zonotope_of_interval(const Interval &I) {
    const Tensor r = I.rad();
    // the axes with a radius (in some member of a batch), by their index
    const Tensor nonzero = r.detach().not_equal(r.zeros_like(r.shape()));
    Tensor any_radius = nonzero;
    while (any_radius.ndim() > 2) any_radius = any_radius.any(0);
    return {I.center(), priv_select_generators(r.diag(), any_radius.nonzero())};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
