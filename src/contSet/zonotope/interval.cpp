// interval - the interval hull of a zonotope, as CORA's interval(Z)
//
// Syntax:   Interval I = Z.interval();
// Outputs:  I - the smallest box containing Z: center c, radius sum_j |G_j|
// See also: Interval

#include "contSet/zonotope/zonotope.h"

namespace cora::ct {

Interval Zonotope::interval() const {
    const Tensor r = G.abs().sumLast();  // the radius: row sums of |G|
    return {c - r, c + r};
}

} // namespace cora::ct
