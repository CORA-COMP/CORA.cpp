// interval - the smallest box that contains an ellipsoid, as CORA's ellipsoid.interval
//
// Syntax:   Interval I = E.interval();
// Inputs:   -
// Outputs:  I - [q - r, q + r] with r_i = sqrt(Q_ii), the support function of the centered
//           ellipsoid along the unit vectors; the empty box for an empty ellipsoid. CORA takes an
//           ellipsoid with Q within eps of zero for a point and gives r = 0
//           (docs/cora-differences.md)
// See also: center, Zonotope::interval
//
// CORA: contSet/@ellipsoid/interval.m

#include "cont_set/ellipsoid/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval Ellipsoid::interval_impl() const {
    if (priv_is_empty(*this)) return Interval::empty(dim());
    const Tensor radius = Q_.diagonal().sqrt();
    return {q_ - radius, radius + q_};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
