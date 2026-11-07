// volume - the volume of an ellipsoid, as CORA's ellipsoid.volume_
//
// The volume of the unit ball of n dimensions times sqrt(det Q); 0 for the empty ellipsoid.
//
// Syntax:   vol = E.volume();
// Inputs:   -
// Outputs:  vol - the volume, (..., 1, 1)
// See also: radius, Capsule::volume
//
// CORA: contSet/@ellipsoid/volume_.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "global/format.h"

#include <cmath>
#include <numbers>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Ellipsoid::volume() const {
    detail::Entry entry;
    if (priv_is_empty(*this)) return q_.zeros_like({1, 1});
    const double n = static_cast<double>(dim());
    const double unit = std::pow(std::numbers::pi, n / 2) / std::tgamma(n / 2 + 1);
    // det Q is the product of the eigenvalues; rounding may leave a tiny negative one
    const Tensor det = Q_.eigh().first.maximum(0.0).prod(static_cast<int>(Q_.ndim()) - 2, true);
    return det.sqrt() * unit;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
