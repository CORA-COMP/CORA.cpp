// priv_axis_parameter - where on the axis of a capsule the points are nearest
//
// Syntax:   Tensor t = priv_axis_parameter(g, d);
// Inputs:   g - the generator (..., n, 1)
//           d - the points relative to the center (..., n, N)
// Outputs:  t - (..., 1, N): d'g / g'g clipped to [-1, 1], the parameter of the nearest point
//           c + t g of the segment c - g to c + g. This is CORA's case split on |proj| < |g|; a
//           zero generator gives 0
// See also: priv_axis_distance, Capsule::enclose_points
//
// CORA: contSet/@capsule/contains_.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/capsule/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor priv_axis_parameter(const Tensor &g, const Tensor &d) {
    const Tensor length2 = g.multiply(g).sum(-2, true);
    const Tensor moving = length2.greater(length2.zeros_like());
    // a zero generator divides by one instead of zero
    const Tensor along = d.multiply(g).sum(-2, true);
    const Tensor foot =
        Tensor::where(moving, along.divide(Tensor::where(moving, length2, length2.ones_like())),
                      along.zeros_like());
    return foot.maximum(-1.0).minimum(1.0);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
