// ellipsoid - the class of ellipsoids, as CORA's ellipsoid
//
// The ellipsoid {x | (x - q)' Q^+ (x - q) <= 1} has the shape matrix Q (..., n, n), symmetric and
// positive semidefinite, and the center q, a column (..., n, 1); a singular Q gives a degenerate
// ellipsoid, Q = 0 a point. Immutable: every operation returns a new set. The empty ellipsoid of n
// dimensions has Q of shape (0, 0) and q of shape (n, 0), as CORA's. The tolerance of the object
// (CORA's TOL) relaxes the symmetry and rank decisions of its operations.
//
// Syntax:     Ellipsoid E(Q, q);   Ellipsoid E(Q);   Ellipsoid::empty(n);
// Operations: constructors (ellipsoid.cpp), empty, origin, center, interval, is_bounded,
//             is_full_dim, generators, ellipsoid_norm, is_bigger, is_bad_dir, enclose_points,
//             intersect_strip, intersect, cart_prod, contains, enclose, support_function,
//             linear_map, plus, mink_diff, project, lift, reduce, norm, isequal, is_intersecting,
//             radius, volume, distance, rand_point, vertices, vertices_3d, zonotope, unite,
//             convex_hull, from_interval, from_zonotope, generate_random, isnan, display (one file
//             each). dim is ContSet's.
// See also:   cora/cont_set/cont_set.h, cora/cont_set/zonotope/zonotope.h
//
// CORA: contSet/@ellipsoid/ellipsoid.m
// CORA: contSet/@ellipsoid/dim.m
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

class Rng;

class Ellipsoid : public ContSet {
  public:
    static constexpr ClassId kClassId = ClassId::ellipsoid;

    /// The ellipsoid of the shape matrix Q (..., n, n) with the center q (..., n, 1), default 0.
    /// tol (CORA's TOL) bounds the asymmetry of Q and its negative eigenvalues, and later the
    /// rank decisions; default: the repository tolerance (docs/architecture.md).
    explicit Ellipsoid(const Tensor &Q, Tol tol = std::nullopt);
    Ellipsoid(Tensor Q, Tensor q, Tol tol = std::nullopt);

    /// The ellipsoid that holds a box (CORA's interval.ellipsoid, mode "outer"), radii widened to
    /// 5e-3; "inner" and a box with a flat edge need an SDP: Error::not_supported.
    explicit Ellipsoid(const Interval &I, const std::string &mode = "outer");

    /// The ellipsoid that holds a zonotope (CORA's zonotope.ellipsoid) of n independent generators,
    /// n G G' for the outer modes, G G' for the inner ones; other zonotopes need an SDP.
    explicit Ellipsoid(const Zonotope &Z, const std::string &mode = "outer:norm_bnd");

    /// The stored shape matrix Q, center q and tolerance (CORA's properties Q, q and TOL).
    const Tensor &Q() const noexcept { return Q_; }
    const Tensor &q() const noexcept { return q_; }
    double tol() const noexcept { return tol_; }

    /// The empty ellipsoid of n dimensions (CORA's ellipsoid.empty): Q without entries (0, 0) and q
    /// of shape (n, 0).
    static Ellipsoid empty(int64_t n = 0);

    /// The ellipsoid that is the origin of R^n, n >= 1 (CORA's ellipsoid.origin): Q = 0, q = 0.
    static Ellipsoid origin(int64_t n);

    /// A random ellipsoid, as CORA's generateRandom: dimension (default that of center, else
    /// 1..30), center (default N(0, 1)^n), is_degenerate (Q singular); Q is scaled to entries
    /// below 10.
    static Ellipsoid generate_random(Rng &rng, std::optional<int64_t> dimension = std::nullopt,
                                     bool is_degenerate = false,
                                     const std::optional<Tensor> &center = std::nullopt);

    using ContSet::rand_point;

    /// N random points (n, N) of one ellipsoid: "standard" (or "uniform", CORA's other name) is
    /// uniform in the volume, "extreme" uniform on the boundary. A point repeats its center.
    Tensor rand_point(int64_t N, Rng &rng, const std::string &type) const;

    // Queries -------------------------------------------------------------------------------------

    /// Always false: an ellipsoid holds numbers (CORA's isnan).
    bool isnan() const;

    /// The semi-axes: the square roots of the i largest eigenvalues of Q (CORA's radius), as
    /// (..., i, 1); the empty ellipsoid gives (n, 0).
    Tensor radius(int64_t i = 1) const;

    /// The volume (..., 1, 1): that of the unit ball times sqrt(det Q); 0 for the empty ellipsoid.
    Tensor volume() const;

    /// Whether the ellipsoids are equal up to tol: the same shape matrices and centers, or both
    /// empty. tol default: the smaller tolerance of the two ellipsoids.
    bool isequal(const Ellipsoid &S, Tol tol = std::nullopt) const;

    /// The distance measure of CORA's distance to the points p (n, N): (x - q)' Q^-1 (x - q) - 1
    /// as (1, N), negative inside; along a null direction of Q a deviation counts 1 / tol of the
    /// ellipsoid. Not a batch.
    Tensor distance(const Tensor &p) const;

    /// Whether the ellipsoid meets the points p (all of them in it, as contains) or an ellipsoid
    /// (CORA's isIntersecting_): exact when one is a point; else the distance of two ellipsoids
    /// needs a solver, Error::not_supported. Not a batch.
    bool is_intersecting(const Tensor &p, Tol tol = std::nullopt) const;
    bool is_intersecting(const Ellipsoid &S, Tol tol = std::nullopt) const;

    /// Always true: the shape matrix is finite.
    bool is_bounded() const;

    /// Whether the shape matrix has full rank n (singular values up to the tolerance times the
    /// largest count as zero); an empty ellipsoid has no interior. For a batch, whether every
    /// member does. tol replaces the tolerance of the object; default: that of the object.
    bool is_full_dim(Tol tol = std::nullopt) const;

    /// A matrix G (..., n, n) with G G' = Q (CORA's generators): U sqrt(D) from the singular value
    /// decomposition Q = U D U'; the empty ellipsoid has a matrix without columns (n, 0).
    Tensor generators() const;

    /// The norm of p (..., n, 1) whose unit ball is the ellipsoid shifted to the origin (CORA's
    /// ellipsoidNorm), sqrt(|p' Q^+ p|), as (..., 1, 1); Inf outside the range of Q.
    Tensor ellipsoid_norm(const Tensor &p) const;

    /// Whether the centered ellipsoid E2 - q2 lies in E - q (CORA's isBigger): the largest
    /// generalized eigenvalue of Q2 against Q is at most 1 + tol. Not a batch.
    bool is_bigger(const Ellipsoid &E2) const;

    /// For every column l of L (n, k): whether l is a bad direction of the Minkowski difference
    /// of this ellipsoid and E2 (CORA's isBadDir). Needs Q positive definite. Not a batch.
    std::vector<bool> is_bad_dir(const Ellipsoid &E2, const Tensor &L) const;

    // Constructing from data ----------------------------------------------------------------------

    /// An ellipsoid that contains the points (n, p) (CORA's enclosePoints): method "cov"
    /// (default) takes the mean as the center and the covariance's axes, scaled to the farthest
    /// point.
    static Ellipsoid enclose_points(const Tensor &points, const std::string &method = "cov");

    // Combining with other sets -------------------------------------------------------------------

    /// The matrices V and bar_V of the measurement model of the method of Liu (2016) for
    /// intersect_strip: V the covariance of the measurement noise (m, m), bar_V its scaling (m, m).
    struct StripMeasurement {
        Tensor V, bar_V;
    };

    /// The ellipsoid and the level sigma_sq that intersect_strip returns.
    using StripResult = std::pair<Ellipsoid, double>;

    /// An ellipsoid that contains the intersection with the strip ||C x - y||_2 <= phi (Gollamudi
    /// 1996; CORA's intersectStrip), and the level sigma_sq of the update. Not a batch.
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

    /// The Cartesian product with an ellipsoid or a point (n2, 1) (CORA's cartProd_), mode
    /// "outer": exact for a point, else the ellipsoid of the product of the two bounding boxes.
    Ellipsoid cart_prod(const Ellipsoid &S, const std::string &mode = "outer") const;
    Ellipsoid cart_prod(const Tensor &p, const std::string &mode = "outer") const;

    /// The Cartesian product of a point p (n1, 1) with an ellipsoid.
    static Ellipsoid cart_prod(const Tensor &p, const Ellipsoid &E,
                               const std::string &mode = "outer");

    /// Whether the ellipsoid contains the points p (n, N), a zonotope or a box, up to tol in the
    /// norm of the ellipsoid (CORA's contains_). Not a batch.
    bool contains(const Tensor &p, const std::string &method = "exact",
                  Tol tol = std::nullopt) const;
    bool contains(const Zonotope &S, const std::string &method = "exact",
                  Tol tol = std::nullopt) const;
    bool contains(const Interval &S, const std::string &method = "exact",
                  Tol tol = std::nullopt) const;
    bool contains(const Ellipsoid &S, const std::string &method = "exact",
                  Tol tol = std::nullopt) const;

    /// The Minkowski sum with a vector, or with an ellipsoid (CORA's plus): the enclosure of Halder
    /// (2018) when both are full-dimensional, exact for points and 1D. Other modes: not_supported.
    Ellipsoid plus(const Tensor &v) const;
    Ellipsoid plus(const Ellipsoid &S, const std::string &mode = "outer:halder") const;

    /// The Minkowski difference with a vector (CORA's minkDiff): the shift by -v; with an ellipsoid
    /// only the cases without directions of the equal-area partition, else Error::not_supported.
    Ellipsoid mink_diff(const Tensor &v) const;
    Ellipsoid mink_diff(const Ellipsoid &S2) const;

    /// M E for a matrix M (..., m, n) (CORA's mtimes): M Q M' (symmetrized) and M q.
    Ellipsoid linear_map(const Tensor &M) const;

    /// The ellipsoid with its coordinates permuted (CORA's lift_, N = dim() and dims a
    /// permutation: an ellipsoid is bounded, so new dimensions cannot be added).
    Ellipsoid lift(int64_t N, const std::vector<int64_t> &dims) const;

    // Unions and conversions ---------------------------------------------------------------------

    /// A set that holds both ellipsoids (CORA's or, mode "outer"), or this one for "inner": an
    /// empty operand gives the other; a point inside the other ellipsoid is absorbed; else the
    /// smallest enclosing ellipsoid needs an SDP, Error::not_supported.
    Ellipsoid unite(const Ellipsoid &S, const std::string &mode = "outer") const;
    Ellipsoid unite(const Tensor &p, const std::string &mode = "outer") const;

    /// CORA's convHull_ of two ellipsoids: unite with mode "outer".
    Ellipsoid convex_hull(const Ellipsoid &S) const;

    /// A parallelotope that holds the ellipsoid ("outer:box") or lies in it ("inner:box"); the norm
    /// modes need the equal-area partition of the sphere: Error::not_supported.
    Zonotope zonotope(const std::string &mode = "outer:box",
                      std::optional<int64_t> number_of_generators = std::nullopt) const;

    // Enclosing ellipsoids ------------------------------------------------------------------------

    /// A set that encloses this ellipsoid and E2 (CORA's enclose): needs an SDP, so
    /// Error::not_supported after the dimensions are checked. With (M, Eplus) it is E2 = M Eplus.
    Ellipsoid enclose(const Ellipsoid &E2) const;
    Ellipsoid enclose(const Tensor &M, const Ellipsoid &Eplus) const;

  protected:
    // the unchecked forms of the ContSet interface
    int64_t dim_impl() const override { return q_.shape()[q_.ndim() - 2]; }
    Tensor center_impl() const override;
    Set reduce_impl(int order) const override;
    Set minkowski_difference_impl(const ContSet &S2,
                                  const std::optional<std::string> &type) const override;
    Set translate_impl(const Tensor &v) const override;
    Set linear_map_impl(const Tensor &M) const override;
    Set lift_impl(int64_t N, const std::vector<int64_t> &proj) const override;
    Tensor norm_impl(Tensor::Norm type, const std::string &mode) const override;
    Tensor support_function_impl(const Tensor &d) const override;
    Interval interval_impl() const override;
    Tensor rand_point_impl(int64_t N, Rng &rng) const override;
    Set project_impl(const std::vector<int64_t> &dims) const override;
    // the plots take the outline and the surface points
    std::vector<Polygon> vertices_impl() const override;
    std::vector<std::vector<Point3>> vertices_3d_impl() const override;
    std::string display_impl() const override;

    bool is_bounded_impl() const override;
    bool is_full_dim_impl(Tol tol) const override;
    Set enclose_impl(const ContSet &S2) const override;

  private:
    Tensor rand_point_impl(int64_t N, Rng &rng, const std::string &type) const;

    Tensor Q_, q_;
    double tol_;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
