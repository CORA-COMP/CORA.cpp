// norm - an upper bound on the norm of the points of a zonotope
//
// Syntax:   val = Z.norm();   val = Z.norm(type, "ub");
// Inputs:   type - 1, 2 (default) or infinity;  mode - "ub" (default): the norm of the box around
//           the zonotope, so no point has a larger norm; "exact" needs a mixed-integer solver and
//           "ub_convex" a semidefinite one, neither is supported
// Outputs:  val - the bound, (..., 1, 1)
// See also: interval, minnorm
//
// CORA: contSet/@zonotope/norm_.m
// CORA: contSet/@interval/norm_.m

#include "cora/cont_set/zonotope/zonotope.h"

#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Zonotope::norm(double type, const std::string &mode) const {
    detail::Entry entry;
    CORA_CHECK(type == 1 || type == 2 || type == std::numeric_limits<double>::infinity(),
               Error::wrong_value, [&] {
                   return "norm: the type must be 1, 2 or infinity, got " + std::to_string(type);
               });
    CORA_CHECK(mode == "exact" || mode == "ub" || mode == "ub_convex", Error::wrong_value, [&] {
        return "norm: the mode must be 'exact', 'ub' or 'ub_convex', got '" + mode + "'";
    });
    CORA_CHECK(mode == "ub", Error::not_supported, [&] {
        return "norm: the mode '" + mode + "' needs a " +
               (mode == "exact" ? "mixed-integer" : "semidefinite") + " solver";
    });
    // the largest magnitude of each coordinate
    const Interval box = interval_impl();
    const Tensor largest = box.inf().abs().maximum(box.sup().abs());
    if (type == 1) return largest.sum(-2, true);
    if (type == 2) return largest.square().sum(-2, true).sqrt();
    return largest.max(-2, true);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
