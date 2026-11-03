// priv_zonotope_of_point - the zonotope of a point
//
// Syntax:   Zonotope Z = priv_zonotope_of_point(p);
// Inputs:   p - a point (..., n, 1)
// Outputs:  Z - center p and the (n, 0) generator matrix
// See also: Zonotope::convex_hull
//
// CORA.cpp only: the conversion is CORA's zonotope(p) of the constructor, which takes a point

#include "cont_set/zonotope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope priv_zonotope_of_point(const Tensor &p) {
    Shape shape = p.shape();
    shape.back() = 0;
    return {p, p.zeros_like(shape)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
