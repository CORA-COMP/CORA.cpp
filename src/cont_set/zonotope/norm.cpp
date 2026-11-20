// norm - an upper bound on the norm of the points of a zonotope
//
// "ub" is the norm of the box around the zonotope, so no point has a larger norm; "exact" needs a
// mixed-integer solver and "ub_convex" a semidefinite one, neither is supported.
//
// Syntax:   val = Z.norm();   val = Z.norm(type, "ub");
// Inputs:   type - 1, 2 (default) or infinity (ContSet::norm also takes fro)
//           mode - "ub" (default)
// Outputs:  val - the bound, (...)
// See also: interval, minnorm
//
// CORA: contSet/@zonotope/norm_.m
// CORA: contSet/@interval/norm_.m
// CORA: matrixSet/@matZonotope/norm.m
// CORA authors: Victor Gassmann
// Authors:       Tobias Ladner, Claude Sonnet 5.5, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

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
    return norm_impl(type == 1   ? Tensor::Norm::one
                     : type == 2 ? Tensor::Norm::two
                                 : Tensor::Norm::inf,
                     mode);
}

Tensor Zonotope::norm_impl(Tensor::Norm type, const std::string &mode) const {
    if (mode != "ub")
        CORA_FAIL(Error::not_supported, "norm: the mode '" + mode + "' needs a " +
                                            (mode == "exact" ? "mixed-integer" : "semidefinite") +
                                            " solver");
    // a matrix zonotope: the norm of its interval matrix (CORA's matZonotope.norm), not the 2-norm
    if (elem_rank_ == 2) {
        if (type == Tensor::Norm::two)
            CORA_FAIL(Error::wrong_value,
                      "norm: the 2-norm of a matrix zonotope is not supported, use 1 or Inf");
        return interval_impl().norm(type);
    }
    // the Frobenius norm of the box, as CORA's norm_ of the interval
    if (type == Tensor::Norm::fro) return interval_impl().norm(type);
    // the largest magnitude of each coordinate
    const Interval box = interval_impl();
    const Tensor largest = box.inf().abs().maximum(box.sup().abs());
    if (type == Tensor::Norm::one) return largest.sum(-1);
    if (type == Tensor::Norm::two) return largest.square().sum(-1).sqrt();
    return largest.max(-1);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
