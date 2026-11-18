// priv_zonotope_of_point - the zonotope of a point
//
// Syntax:   Zonotope Z = priv_zonotope_of_point(p);
// Inputs:   p - a point (..., n, 1)
// Outputs:  Z - center p and the (n, 0) generator matrix
// See also: Zonotope::convex_hull
//
// CORA.cpp only: the conversion is CORA's zonotope(p) of the constructor, which takes a point
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cont_set/zonotope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope priv_zonotope_of_point(const Tensor &p) {
    // the point is a column (..., n, 1)
    Shape shape = p.shape();
    shape[shape.size() - 2] = 0;
    shape.back() = p.size(-2);
    return {p.squeeze(-1), p.zeros_like(shape)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
