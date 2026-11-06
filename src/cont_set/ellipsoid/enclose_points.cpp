// enclose_points - an ellipsoid that contains a point cloud, as CORA's ellipsoid.enclosePoints
//
// A degenerate cloud gives a degenerate ellipsoid. In one dimension the ellipsoid is the range;
// in more, the axes are the singular vectors of the covariance and the radii the largest
// coordinates, scaled so that the farthest point is on the boundary. Points that all coincide
// give that point, where CORA raises an error (docs/cora-differences.md). "min-vol" (the minimum-
// volume ellipsoid, an SDP) is Error::not_supported.
//
// Syntax:   Ellipsoid E = Ellipsoid::enclose_points(points);   Ellipsoid::enclose_points(points,
// "cov"); Inputs:   points - the points as columns (n, p), at least one; method - "cov" (default)
// or
//           "min-vol"
// Outputs:  E - the mean as the center, the points rotated onto the subspace they span
// See also: Interval::enclose_points, Zonotope::enclose_points
//
// CORA: contSet/@ellipsoid/enclosePoints.m
// CORA authors: Niklas Kochdumper, Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The shape matrix and center of the ellipsoid of the centered points (k, p), k >= 1 of them
/// spanning R^k, in the same coordinates; method "cov" only.
std::pair<Tensor, Tensor> aux_reduced(const Tensor &points, const std::string &method) {
    const int64_t k = points.shape()[0], count = points.shape()[1];
    const Tensor origin = points.zeros_like({k, 1});
    if (k == 1) {
        // the range of the points is exact in one dimension
        const Tensor high = points.max(1, true), low = points.min(1, true);
        const Tensor radius = (high - low) * 0.5;
        return {radius.multiply(radius), (high + low) * 0.5};
    }
    if (method != "cov")
        CORA_FAIL(Error::not_supported,
                  "enclose_points: the method '" + method + "' needs an SDP solver");
    // the sample covariance of the points, normalized by p - 1
    const Tensor deviations = points - points.mean(1, true);
    const Tensor covariance = deviations.matmul(deviations.matrix_transpose()) / double(count - 1);
    const Tensor U = covariance.svd()[0];
    const Tensor oriented = U.matrix_transpose().matmul(points);
    // the radius of each axis is its largest coordinate, as CORA takes it
    const Tensor radius = oriented.max(1, true).maximum(oriented.min(1, true));
    const Tensor scaled = oriented.square().divide(radius.square()).sum(0, true);
    const Tensor B = radius.square().diag() * scaled.max().item();
    return {priv_congruence(U, B), origin};
}

} // namespace


// ===========================================  MAIN  =========================================== //

Ellipsoid Ellipsoid::enclose_points(const Tensor &points, const std::string &method) {
    detail::Entry entry;
    CORA_CHECK(
        points.defined() && points.ndim() == 2 && points.numel() > 0, Error::wrong_value,
        [] { return std::string("enclose_points: points must be a nonempty matrix (n, p)"); });
    CORA_CHECK(method == "cov" || method == "min-vol", Error::wrong_value, [&] {
        return "enclose_points: the method must be 'cov' or 'min-vol', got '" + method + "'";
    });
    const int64_t n = points.shape()[0], count = points.shape()[1];
    const Tensor c = points.mean(1, true);
    const Tensor centered = points - c;
    // the padded cloud has a full rotation T, even for fewer points than dimensions
    const Tensor padded =
        count < n ? Tensor::concat({centered, centered.zeros_like({n, n - count})}, 1) : centered;
    const Tensor T = padded.svd()[0];
    const int64_t rank = padded.matrix_rank();
    // a cloud of one point is that point
    if (rank == 0) return {points.zeros_like({n, n}), c};
    const auto [Q_reduced, q_reduced] =
        aux_reduced(T.matrix_transpose().matmul(centered).slice(0, 0, rank), method);
    const int64_t missing = n - rank;
    // back in the original space: the directions without spread get no extent
    const Tensor Q_full =
        missing > 0 ? Tensor::block_diag({Q_reduced, points.zeros_like({missing, missing})})
                    : Q_reduced;
    const Tensor q_full =
        missing > 0 ? Tensor::concat({q_reduced, points.zeros_like({missing, 1})}, 0) : q_reduced;
    return {priv_congruence(T, Q_full), T.matmul(q_full) + c};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
