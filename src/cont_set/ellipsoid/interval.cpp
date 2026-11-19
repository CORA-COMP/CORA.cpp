// interval - the smallest box that contains an ellipsoid, as CORA's ellipsoid.interval
//
// The empty ellipsoid gives the empty box. CORA takes an ellipsoid with Q within eps of zero for
// a point and gives r = 0 (docs/cora-differences.md).
//
// Syntax:   Interval I = E.interval();
// Inputs:   -
// Outputs:  I - [q - r, q + r] with r_i = sqrt(Q_ii), the support function along the unit vectors
// See also: center, Zonotope::interval
//
// CORA: contSet/@ellipsoid/interval.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Ellipsoid::interval_impl() const {
    if (priv_is_empty(*this)) return Interval::empty(dim());
    const Tensor radius = Q_.diagonal().sqrt();
    return {(q_ - radius).squeeze(-1), (radius + q_).squeeze(-1)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
