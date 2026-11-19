// distance - the distance measure of points to an ellipsoid, as CORA's ellipsoid.distance
//
// (x - q)' Q^-1 (x - q) - 1 for each point: negative inside, 0 on the boundary. In a null direction
// of Q a deviation counts 1 / tol, the tolerance of the ellipsoid. The distances to ellipsoids and
// polytopes need a solver and are not ported.
//
// Syntax:   d = E.distance(p);
// Inputs:   p - points (..., n), one per row
// Outputs:  d - (...), one value per point
// See also: is_intersecting, ellipsoid_norm
//
// CORA: contSet/@ellipsoid/distance.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       06-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Ellipsoid::distance(const Tensor &p) const {
    detail::Entry entry;
    CORA_CHECK(p.defined() && is_vector(p, dim()), Error::dimension_mismatch, [&] {
        return "distance: the points must be (..., " + std::to_string(dim()) +
               "), one per row, got the shape " + (p.defined() ? shape_text(p.shape()) : "()");
    });
    CORA_CHECK(!priv_is_empty(*this), Error::empty_set,
               [] { return std::string("distance: the ellipsoid is empty"); });
    CORA_CHECK(Q_.ndim() == 2, Error::not_supported,
               [] { return std::string("distance: it supports a single ellipsoid, not a batch"); });
    // in the basis of Q, the squared deviations weigh 1 / s_i, or 1 / tol where s_i counts as 0
    const std::array<Tensor, 3> usv = Q_.svd();
    const Tensor &s = usv[1];
    const Tensor kept =
        s.greater(s.zeros_like()).multiply(s.greater_equal(s.slice(0, 0, 1) * tol_));
    const Tensor weight =
        Tensor::where(kept, s.ones_like().divide(Tensor::where(kept, s, s.ones_like())),
                      s.full_like(s.shape(), 1.0 / tol_));
    // the points as CORA's columns (n, N), the measures back in the shape of the points
    const Shape batch(std::vector<int64_t>(p.shape().begin(), p.shape().end() - 1));
    const Tensor P = p.reshape({p.numel() / dim(), dim()}).matrix_transpose();
    const Tensor w = usv[0].matrix_transpose().matmul(P - q_.expand_dims(-1));
    return (w.square().multiply(weight).sum(0, true) - 1.0).reshape(batch);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
