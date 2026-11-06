// priv_segment_distance - the least distance between the axes of two capsules
//
// Syntax:   Tensor dist = priv_segment_distance(c1, g1, c2, g2);
// Inputs:   c1, g1, c2, g2 - the centers and generators (..., n, 1)
// Outputs:  dist - (..., 1, 1): min |c1 + g1 x1 - c2 - g2 x2| over x1, x2 in [-1, 1]. CORA asks a
//           quadratic program for the minimizer; the problem has two variables, so the minimizer
//           is the stationary point of the squared distance if it lies in the box, else on one of
//           the four edges, where one parameter is fixed and the other is the clipped minimizer.
//           Every candidate is clipped into the box, so each is a pair of points on the segments
// See also: priv_axis_distance, Capsule::is_intersecting
//
// CORA: contSet/@capsule/isIntersecting_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/capsule/private/priv.h"

#include <utility>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The entries of the parameters limited to the box [-1, 1].
Tensor aux_clip(const Tensor &x) { return x.maximum(-1.0).minimum(1.0); }

/// numerator / denominator where the denominator is positive, else 0.
Tensor aux_divide(const Tensor &numerator, const Tensor &denominator) {
    const Tensor positive = denominator.greater(denominator.zeros_like());
    return Tensor::where(
        positive, numerator.divide(Tensor::where(positive, denominator, denominator.ones_like())),
        numerator.zeros_like());
}

} // namespace


// ===========================================  MAIN  =========================================== //

Tensor priv_segment_distance(const Tensor &c1, const Tensor &g1, const Tensor &c2,
                             const Tensor &g2) {
    const auto dot = [](const Tensor &a, const Tensor &b) { return a.multiply(b).sum(-2, true); };
    const Tensor d = c1.subtract(c2);
    const Tensor a = dot(g1, g1), b = dot(g1, g2), e = dot(g2, g2), p1 = dot(d, g1),
                 p2 = dot(d, g2);
    // the stationary point of |d + g1 x1 - g2 x2|^2: a x1 - b x2 = -p1 and -b x1 + e x2 = p2
    const Tensor det = a.multiply(e).subtract(b.multiply(b));
    const Tensor inner1 = aux_clip(aux_divide(b.multiply(p2).subtract(e.multiply(p1)), det));
    const Tensor inner2 = aux_clip(aux_divide(a.multiply(p2).subtract(b.multiply(p1)), det));
    // x2 for a given x1 and x1 for a given x2, each the minimizer along its edge
    const auto along2 = [&](const Tensor &x1) {
        return aux_clip(aux_divide(p2.add(b.multiply(x1)), e));
    };
    const auto along1 = [&](const Tensor &x2) {
        return aux_clip(aux_divide(b.multiply(x2).subtract(p1), a));
    };
    const Tensor low = a.full_like(a.shape(), -1.0), high = a.ones_like();
    const std::vector<std::pair<Tensor, Tensor>> candidates{{inner1, inner2},
                                                            {low, along2(low)},
                                                            {high, along2(high)},
                                                            {along1(low), low},
                                                            {along1(high), high}};
    Tensor nearest;
    for (const auto &[x1, x2] : candidates) {
        const Tensor distance = priv_norm(d.add(g1.multiply(x1)).subtract(g2.multiply(x2)));
        nearest = nearest.defined() ? nearest.minimum(distance) : distance;
    }
    return nearest;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
