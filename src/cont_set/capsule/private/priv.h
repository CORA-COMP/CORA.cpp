// priv - the private functions of capsule, as CORA's @capsule/private
//
// Not for callers of the class: the operations of the capsule use them.
//
// Syntax:   Tensor d = priv_axis_distance(c, g, p);   bool ok = priv_all(priv_at_most(a, b, tol));
// See also: capsule.h
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/capsule/capsule.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// Why p is no matrix of points (..., n, N) of the dimension n of C, if it is none.
detail::Refusal priv_points_refusal(const Capsule &C, const Tensor &p);

/// 1 for the members whose radius r (..., 1, 1) is negative (the empty ones), else 0.
Tensor priv_empty_mask(const Tensor &r);

/// Whether every entry of a 0/1 mask is 1 (true for a mask without entries).
bool priv_all(const Tensor &mask);

/// 1 where a <= b or a and b agree up to tol (as within_tol), else 0.
Tensor priv_at_most(const Tensor &a, const Tensor &b, double tol);

/// The Euclidean length of the columns of x (..., n, N): (..., 1, N).
Tensor priv_norm(const Tensor &x);

/// The parameter t in [-1, 1] of the point c + t g nearest to the points c + d (g (..., n, 1),
/// d (..., n, N)): (..., 1, N), the foot of the perpendicular d'g / g'g clipped to the segment;
/// 0 for a zero generator.
Tensor priv_axis_parameter(const Tensor &g, const Tensor &d);

/// The distance of the points p (..., n, N) from the segment c - g to c + g (c and g (..., n, 1)):
/// (..., 1, N). The foot of the perpendicular is clipped to the segment; a zero generator is the
/// distance from c.
Tensor priv_axis_distance(const Tensor &c, const Tensor &g, const Tensor &p);

/// The least distance between the segments c1 - g1 .. c1 + g1 and c2 - g2 .. c2 + g2: (..., 1, 1).
/// The minimum of the squared distance over the parameters in [-1, 1]^2 is at the stationary
/// point or on an edge of the box, where each parameter is the clipped minimizer of the other.
Tensor priv_segment_distance(const Tensor &c1, const Tensor &g1, const Tensor &c2,
                             const Tensor &g2);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
