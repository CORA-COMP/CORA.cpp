// support_function - the support function of an ellipsoid, as CORA's ellipsoid.supportFunc_
//
// The largest value of d'x over the ellipsoid: d'q + sqrt(d'Qd). A point (Q = 0 up to eps) gives
// d'q, and the empty ellipsoid -Inf.
//
// Syntax:   val = E.support_function(d);
// Inputs:   d - direction (..., n)
// Outputs:  val - the maximum, (...)
// See also: interval
//
// CORA: contSet/@ellipsoid/supportFunc_.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       06-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "cora/global/within_tol.h"
#include "global/format.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Ellipsoid::support_function_impl(const Tensor &direction) const {
    // the products of CORA's column of the direction, (..., 1, 1), then (...)
    const Tensor d = direction.expand_dims(-1);
    Shape shape = direction.shape();
    shape.pop_back();
    if (priv_is_empty(*this)) return d.full_like(shape, -std::numeric_limits<double>::infinity());
    const Tensor dt = d.matrix_transpose();
    const Tensor shift = dt.matmul(q_.expand_dims(-1)), quadratic = dt.matmul(Q_).matmul(d);
    // a point has no spread; the root of the placeholder 1 keeps its gradient finite
    const Tensor point =
        within_tol(Q_, Q_.zeros_like(), machine_eps(Q_.dtype())).all(-1, true).all(-2, true);
    const Tensor spread =
        Tensor::where(point, quadratic.zeros_like(), quadratic.maximum(0.0).sqrt());
    return shift.add(spread).squeeze(-1).squeeze(-1);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
