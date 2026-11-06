// volume - the volume of a capsule, as CORA's capsule.volume_
//
// The ball of radius r plus the cylinder over the (n-1)-ball of length 2 |g|; 0 for the empty
// capsule.
//
// Syntax:   vol = C.volume();
// Inputs:   -
// Outputs:  vol - the volume, (..., 1, 1)
// See also: radius
//
// CORA: contSet/@capsule/volume_.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/capsule/private/priv.h"
#include "cora/cont_set/capsule/capsule.h"
#include "global/format.h"

#include <cmath>
#include <numbers>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Capsule::volume() const {
    detail::Entry entry;
    const double n = static_cast<double>(dim());
    // the volumes of the unit balls of n and n - 1 dimensions
    const double unit = std::pow(std::numbers::pi, n / 2) / std::tgamma(n / 2 + 1);
    const double unit_less = std::pow(std::numbers::pi, (n - 1) / 2) / std::tgamma((n - 1) / 2 + 1);
    // the clipped radius keeps the empty member's marker out of the powers (and their gradients)
    const Tensor r = r_.maximum(0.0);
    const Tensor ball = r.pow(n) * unit,
                 cylinder = (r.pow(n - 1) * unit_less).multiply(priv_norm(g_) * 2.0);
    return Tensor::where(priv_empty_mask(r_), r_.zeros_like(), ball.add(cylinder));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
