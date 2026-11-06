// ellipsoid - the class of ellipsoids, as CORA's ellipsoid
//
// The ellipsoid {x | (x - q)' Q^+ (x - q) <= 1} has the shape matrix Q (..., n, n), symmetric and
// positive semidefinite, and the center q, a column (..., n, 1); a singular Q gives a degenerate
// ellipsoid, Q = 0 a point. Immutable: every operation returns a new set. The empty ellipsoid of n
// dimensions has Q of shape (0, 0) and q of shape (n, 0), as CORA's. The tolerance of the object
// (CORA's TOL) relaxes the symmetry and rank decisions of its operations.
//
// Syntax:     Ellipsoid E(Q, q);   Ellipsoid E(Q);   Ellipsoid::empty(n);
// Operations: constructors (ellipsoid.cpp), empty, center, interval, is_bounded, is_full_dim,
//             generators, ellipsoid_norm, is_bigger, is_bad_dir, enclose_points, intersect_strip,
//             intersect, cart_prod, contains, enclose, display (one file each). The other
//             operations of the ContSet interface (support_function, rand_point, project, vertices)
//             are Error::not_supported until they are ported.
// See also:   cora/cont_set/cont_set.h, cora/cont_set/zonotope/zonotope.h
//
// CORA: contSet/@ellipsoid/ellipsoid.m
// CORA authors: Victor Gassmann, Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"
#include "cora/cont_set/zonotope/zonotope.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

class Ellipsoid : public ContSet {
  public:
    static constexpr ClassId kClassId = ClassId::ellipsoid;

    /// The ellipsoid of the shape matrix Q (..., n, n) with the center q (..., n, 1), default 0.
    /// tol (CORA's TOL) bounds the asymmetry of Q and its negative eigenvalues, and later the
    /// rank decisions; default: the repository tolerance (docs/architecture.md).
    explicit Ellipsoid(const Tensor &Q, Tol tol = std::nullopt);
    Ellipsoid(Tensor Q, Tensor q, Tol tol = std::nullopt);

    /// The stored shape matrix Q, center q and tolerance (CORA's properties Q, q and TOL).
    const Tensor &Q() const noexcept { return Q_; }
    const Tensor &q() const noexcept { return q_; }
    double tol() const noexcept { return tol_; }

    /// The empty ellipsoid of n dimensions (CORA's ellipsoid.empty): Q without entries (0, 0) and q
    /// of shape (n, 0).
    static Ellipsoid empty(int64_t n = 0);

    // Queries -------------------------------------------------------------------------------------

    /// Always true: the shape matrix is finite.
    bool is_bounded() const;

    /// Whether the shape matrix has full rank n (singular values up to the tolerance times the
    /// largest count as zero); an empty ellipsoid has no interior. For a batch, whether every
    /// member does. tol replaces the tolerance of the object; default: that of the object.
    bool is_full_dim(Tol tol = std::nullopt) const;

    /// A matrix G (..., n, n) with G G' = Q (CORA's generators): U sqrt(D) from the singular value
    /// decomposition Q = U D U'; the empty ellipsoid has a matrix without columns (n, 0).
    Tensor generators() const;

    /// The norm of p (..., n, 1) whose unit ball is the ellipsoid shifted to the origin
    /// (CORA's ellipsoidNorm), sqrt(|p' Q^+ p|), as (..., 1, 1): Q \ p for a full-rank Q,
    /// else through the pseudo-inverse of the singular value decomposition, with singular values
    /// and entries below the fixed threshold of the method set to zero; Inf for a direction
    /// outside the range of Q.
    Tensor ellipsoid_norm(const Tensor &p) const;

    /// Whether the centered ellipsoid E2 - q2 lies in E - q (CORA's isBigger): the largest
    /// generalized eigenvalue of Q2 against Q is at most 1 + tol, tol the least tolerance of the
    /// two; ellipsoids that are not full-dimensional are compared in the subspace that Q + Q2
    /// spans. CORA answers true for every pair of dimension 1 before comparing anything, which is
    /// wrong (docs/cora-differences.md); here the comparison is made. Not a batch; Error::empty_set
    /// for an empty one.
    bool is_bigger(const Ellipsoid &E2) const;

    /// For every column l of L (n, k): whether the direction l is a bad direction of the Minkowski
    /// difference of this ellipsoid and E2 (CORA's isBadDir), sqrt(l' Q l) / sqrt(l' Q2 l) above
    /// 1 / lambda + tol for the largest generalized eigenvalue lambda of Q2 against Q, tol the
    /// least tolerance of the two. Needs Q positive definite. Not a batch.
    std::vector<bool> is_bad_dir(const Ellipsoid &E2, const Tensor &L) const;

    // Constructing from data ----------------------------------------------------------------------

    /// An ellipsoid that contains the points (n, p) (CORA's enclosePoints): method "cov"
    /// (default) takes the mean as the center and the singular value decomposition of the
    /// covariance as the axes, scaled to the farthest point. The points are first rotated onto the
    /// subspace they span, so a degenerate cloud gives a degenerate ellipsoid; points that all
    /// coincide give that point (CORA raises an error). "min-vol" (an SDP) is Error::not_supported.
    static Ellipsoid enclose_points(const Tensor &points, const std::string &method = "cov");

    // Combining with other sets -------------------------------------------------------------------

    /// The matrices V and bar_V of the measurement model of the method of Liu (2016) for
    /// intersect_strip: V the covariance of the measurement noise (m, m), bar_V its scaling (m, m).
    struct StripMeasurement {
        Tensor V, bar_V;
    };

    /// The ellipsoid and the level sigma_sq that intersect_strip returns.
    using StripResult = std::pair<Ellipsoid, double>;

    /// An ellipsoid that contains the intersection with the strip ||C x - y||_2 <= phi (C (m, n),
    /// y (m, 1), phi a number) by the method of Gollamudi (1996) (CORA's intersectStrip). The
    /// second result is the level sigma_sq of the update, to be passed as sigma_sq_prev to the
    /// next one. CORA's shape and center are too small (docs/cora-differences.md); here the
    /// shape is sigma_sq times the inverse of the information matrix, and the center is moved
    /// with that inverse. The update is skipped (the ellipsoid and sigma_sq_prev are returned)
    /// when the strip does not cut it or the weight leaves [0, 1). Needs Q positive definite. Not a
    /// batch.
    StripResult intersect_strip(const Tensor &C, double phi, const Tensor &y,
                                double sigma_sq_prev = 0.0) const;

    /// The same for the method of Liu (2016) with the measurement model sys (CORA's struct with the
    /// fields V and bar_V); the measurement set is (y - C x)' V^-1 (y - C x) <= 1. The shape is
    /// corrected as above (docs/cora-differences.md).
    StripResult intersect_strip(const Tensor &C, const StripMeasurement &sys, const Tensor &y,
                                double sigma_sq_prev = 0.0) const;

    /// The intersection with a point p (n, 1) (CORA's and_): the point as an ellipsoid with Q = 0
    /// if the ellipsoid contains it up to the machine epsilon, else the empty ellipsoid.
    Ellipsoid intersect(const Tensor &p) const;

    /// The intersection with an ellipsoid: mode "outer" (default) or "inner". An ellipsoid that is
    /// a point is intersected as such; the other cases need the intersection tests and solvers of
    /// the polytope and ellipsoid classes and are Error::not_supported. Not a batch.
    Ellipsoid intersect(const Ellipsoid &S, const std::string &mode = "outer") const;

    /// The Cartesian product with an ellipsoid or a point (n2, 1) (CORA's cartProd_), mode "outer"
    /// ("inner" and "exact" are Error::not_supported, as in CORA). With a point it is exact: the
    /// center is stacked with the point and Q padded with zeros. With an ellipsoid it is the
    /// ellipsoid of the product of the two bounding boxes, each box first widened to a radius of
    /// at least 5e-3 (as CORA's interval-to-ellipsoid conversion does); a box with a zero radius
    /// is Error::not_supported (CORA calls a solver).
    Ellipsoid cart_prod(const Ellipsoid &S, const std::string &mode = "outer") const;
    Ellipsoid cart_prod(const Tensor &p, const std::string &mode = "outer") const;

    /// The Cartesian product of a point p (n1, 1) with an ellipsoid.
    static Ellipsoid cart_prod(const Tensor &p, const Ellipsoid &E,
                               const std::string &mode = "outer");

    /// Whether the ellipsoid contains the points p (n, N) (all of them), a zonotope or a box, up to
    /// tol (default: the repository tolerance) in the norm of the ellipsoid (CORA's contains_).
    /// method "exact": the vertices of the set are the points; "approx" (an SDP) is
    /// Error::not_supported. An ellipsoid as the contained set also needs an SDP and is
    /// Error::not_supported. Not a batch.
    bool contains(const Tensor &p, const std::string &method = "exact",
                  Tol tol = std::nullopt) const;
    bool contains(const Zonotope &S, const std::string &method = "exact",
                  Tol tol = std::nullopt) const;
    bool contains(const Interval &S, const std::string &method = "exact",
                  Tol tol = std::nullopt) const;
    bool contains(const Ellipsoid &S, const std::string &method = "exact",
                  Tol tol = std::nullopt) const;

    /// A set that encloses this ellipsoid and E2 (CORA's enclose): the convex hull of this
    /// ellipsoid and the ellipsoid E2 = M Eplus, which needs the SDP of the outer union and is
    /// Error::not_supported after the dimensions are checked. With (M, Eplus), M (n, m) and
    /// Eplus of dimension m. CORA's documentation says M E + Eplus, its code takes M Eplus
    /// (docs/cora-differences.md); the code's form is kept.
    Ellipsoid enclose(const Ellipsoid &E2) const;
    Ellipsoid enclose(const Tensor &M, const Ellipsoid &Eplus) const;

  protected:
    int64_t dim_impl() const override { return q_.shape()[q_.ndim() - 2]; }
    Tensor center_impl() const override;
    Tensor support_function_impl(const Tensor &d) const override;
    Interval interval_impl() const override;
    Tensor rand_point_impl(int64_t N, Rng &rng) const override;
    Set project_impl(const std::vector<int64_t> &dims) const override;
    std::vector<Polygon> vertices_impl() const override;
    std::vector<std::vector<Point3>> vertices_3d_impl() const override;
    std::string display_impl() const override;

    bool is_bounded_impl() const override;
    bool is_full_dim_impl(Tol tol) const override;
    Set enclose_impl(const ContSet &S2) const override;

  private:
    Tensor Q_, q_;
    double tol_;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
