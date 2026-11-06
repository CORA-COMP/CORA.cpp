// norm - an upper bound of the 2-norm of the points of an ellipsoid, as CORA's ellipsoid.norm_
//
// For an ellipsoid around the origin the largest norm is the root of the largest eigenvalue of Q;
// the empty ellipsoid has -Inf. CORA refuses other norms and a center other than 0.
//
// Syntax:   val = E.norm();
// Inputs:   -
// Outputs:  val - the largest 2-norm, (..., 1, 1)
// See also: radius, ContSet::norm
//
// CORA: contSet/@ellipsoid/norm_.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "global/format.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Ellipsoid::norm_impl(Tensor::Norm type, const std::string &) const {
    if (type != Tensor::Norm::two)
        CORA_FAIL(Error::no_exact_alg, "norm: the ellipsoid has only the Euclidean norm");
    if (priv_is_empty(*this)) return q_.full_like({1, 1}, -std::numeric_limits<double>::infinity());
    if (q_.abs().max().item() != 0.0)
        CORA_FAIL(Error::not_supported, "norm: not yet implemented for a nonzero center");
    // the eigenvalues ascend, so the last is the largest
    return Q_.eigh().first.max(static_cast<int>(Q_.ndim()) - 2, true).sqrt();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
