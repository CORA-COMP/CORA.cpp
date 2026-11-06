// priv_axis_distance - the distance of points from the axis of a capsule
//
// Syntax:   Tensor dist = priv_axis_distance(c, g, p);
// Inputs:   c, g - the center and the generator (..., n, 1)
//           p - the points (..., n, N)
// Outputs:  dist - (..., 1, N): the distance of each point from the segment c - g to c + g.
//           The foot of the perpendicular, c + t g with t = (p - c)'g / g'g, is clipped to
//           t in [-1, 1], which is CORA's case split on |proj| < |g| (the distance to the axis
//           or to the nearer end); a zero generator gives t = 0
// See also: priv_at_most, Capsule::contains
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

Tensor priv_axis_distance(const Tensor &c, const Tensor &g, const Tensor &p) {
    const Tensor t = priv_axis_parameter(g, p.subtract(c));
    return priv_norm(c.add(t.multiply(g)).subtract(p));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
