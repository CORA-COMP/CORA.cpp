// ball_radius - the radius of the ball around the center that holds an ellipsoid
//
// The largest semi-axis: the root of the largest eigenvalue of Q, as Interval::ball_radius and
// Zonotope::ball_radius are the radius of their enclosing ball (#163).
//
// Syntax:   r = E.ball_radius();
// Inputs:   -
// Outputs:  r - the radius (...); 0 for the empty ellipsoid
// See also: semi_axes, Interval::ball_radius, Zonotope::ball_radius
//
// CORA.cpp only: CORA's ellipsoid.radius is the semi-axes (semi_axes); the other sets' radius is
// this ball's.
// Authors:       Claude Opus 5.5
// Written:       19-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Ellipsoid::ball_radius() const {
    detail::Entry entry;
    const Shape batch(std::vector<int64_t>(q_.shape().begin(), q_.shape().end() - 1));
    if (priv_is_empty(*this)) return q_.zeros_like(batch);
    // the eigenvalues ascend, (..., n, 1): the last is the largest
    const Tensor values = Q_.eigh().first;
    const int axis = static_cast<int>(values.ndim()) - 2;
    return values.slice(axis, dim() - 1, dim()).maximum(0.0).sqrt().reshape(batch);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
