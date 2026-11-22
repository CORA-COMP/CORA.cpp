// capsule - the class of capsules, as CORA's capsule
//
// A capsule is the set of points within the radius r of the segment c - g to c + g: C = {x |
// min_{t in [-1, 1]} |x - (c + t g)|_2 <= r}. Stored batch first as Zonotope (#163): the center c
// and the generator g are (..., n), the radius r is (...) and nonnegative; r = 0 is a segment,
// g = 0 a ball. Points and directions are (..., n), N points (..., N, n), as for every set.
// Immutable: every operation returns a new set. The empty members are flagged in empty_mask(), as
// for every set; empty(n) is such a set.
//
// Syntax:     Capsule C(c, g, r);   Capsule C(c);   Capsule C(I);   Capsule C(Z);
//             Capsule::empty(n);   Capsule::origin(n);   Capsule::stack({C1, C2});
// Operators:  M * C (linear_map)
// Operations: constructors (capsule.cpp), empty, origin, stack, enclose_points, center, interval,
//             is_bounded, is_full_dim (the ContSet entries), isequal, contains, is_intersecting,
//             convex_hull, enclose, linear_map, lift, plus, ball_radius, volume, representsa,
//             zonotope, ellipsoid, polytope, generate_random, isnan, support_function, rand_point,
//             project, vertices, vertices_3d, reduce, display (one file each). dim is ContSet's.
// See also:   cora/cont_set/cont_set.h, cora/cont_set/zonotope/zonotope.h
//
// CORA: contSet/@capsule/capsule.m
// CORA: contSet/@capsule/dim.m
// CORA: contSet/@capsule/isnan.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       06-November-2026
// Last update:   20-November-2026
// Reviewed:      ---

#pragma once

#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"
#include "cora/cont_set/zonotope/zonotope.h"

#include <optional>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

class Ellipsoid;
class Polytope;
class Rng;

class Capsule : public ContSet {
  public:
    static constexpr ClassId kClassId = ClassId::capsule;

    /// The capsule of the center c (..., n), the generator g (..., n; default 0) and the radius r
    /// (...) >= 0 (default 0; a number is the radius of every member).
    explicit Capsule(const Tensor &c);
    Capsule(const Tensor &c, const Tensor &g);
    Capsule(const Tensor &c, const Tensor &g, Tensor r);
    Capsule(const Tensor &c, const Tensor &g, double r);

    /// The capsule that holds a box (CORA's interval.capsule): the segment runs along a widest
    /// edge through the center and the ball covers the half-widths of the other edges.
    explicit Capsule(const Interval &I);

    /// The capsule that holds a zonotope (CORA's zonotope.capsule): the box of the zonotope in the
    /// basis of the left singular vectors of [-G, G], as a capsule, rotated back. Not a batch.
    explicit Capsule(const Zonotope &Z);

    /// The stored center c (..., n), generator g (..., n) and radius r (...); the data of an empty
    /// member is not read.
    const Tensor &c() const noexcept { return c_; }
    const Tensor &generator() const noexcept { return g_; }
    const Tensor &r() const noexcept { return r_; }

    /// The empty capsule of n >= 0 dimensions (CORA's capsule.empty): flagged in empty_mask(), with
    /// the center and generator 0 (n) and the radius 0 (CORA's has a center without columns).
    static Capsule empty(int64_t n = 0);

    /// The capsule that is the origin of R^n, n >= 1 (CORA's capsule.origin).
    static Capsule origin(int64_t n);

    /// The capsules of one shape as one batch along a new first axis, each member empty where its
    /// capsule is.
    static Capsule stack(const std::vector<Capsule> &Cs);

    /// Whether the capsule is empty (flagged in empty_mask()); for a batch, whether every member
    /// is.
    bool is_empty() const;

    /// A capsule that contains the points (..., N, n) (CORA's enclosePoints): the mean as center,
    /// the generator to the farthest point, the largest distance from the segment as radius.
    static Capsule enclose_points(const Tensor &rows);

    /// A random capsule, as CORA's generateRandom: dimension (default that of center, else 1..10),
    /// center (default 5 N(0, 1)), radius (default U[0, 5]); the generator is U[0, 1]^n.
    static Capsule generate_random(Rng &rng, std::optional<int64_t> dimension = std::nullopt,
                                   const std::optional<Tensor> &center = std::nullopt,
                                   std::optional<double> radius = std::nullopt);

    using ContSet::rand_point;

    /// N random points (..., N, n): type "standard" (default of rand_point) draws a point of the
    /// axis plus a point of the ball; "extreme" the point of the boundary in a random direction.
    Tensor rand_point(int64_t N, Rng &rng, const std::string &type) const;

    // Queries -------------------------------------------------------------------------------------

    /// Always false: a capsule holds numbers (CORA's isnan).
    bool isnan() const;

    /// The radius of a ball around the center that holds the capsule (CORA's radius): |g| + r, as
    /// (...).
    Tensor ball_radius() const;

    /// The volume (...): the ball of radius r plus the cylinder of length 2 |g| and radius r
    /// over the (n-1)-ball; 0 for the empty capsule.
    Tensor volume() const;

    /// Whether the capsule is a set of the class `type` (CORA's representsa_, the types it
    /// answers without a conversion), up to tol. Not a batch.
    bool representsa(const std::string &type, Tol tol = std::nullopt) const;

    /// Whether the capsule equals S: the centers, generators and radii agree as within_tol, or
    /// both are empty; a different dimension is false. For a batch, whether every member does.
    /// tol default: the repository tolerance.
    bool isequal(const Capsule &S, Tol tol = std::nullopt) const;

    /// Whether the capsule contains the points p (..., N, n), a capsule, a zonotope or a box, up
    /// to tol in the distance to the axis; for a batch, every member.
    bool contains(const Tensor &p, Tol tol = std::nullopt) const;
    bool contains(const Capsule &S, Tol tol = std::nullopt) const;
    bool contains(const Zonotope &Z, Tol tol = std::nullopt) const;
    bool contains(const Interval &I, Tol tol = std::nullopt) const;

    /// Whether the capsule meets the points p (all of them in it) or a capsule: the least
    /// distance of the two axes is at most the sum of the radii, up to tol. Exact.
    bool is_intersecting(const Tensor &p, Tol tol = std::nullopt) const;
    bool is_intersecting(const Capsule &S, Tol tol = std::nullopt) const;

    // Combining with other sets -------------------------------------------------------------------

    /// A capsule that encloses this one and S (CORA's convHull_ of two capsules): the axis runs
    /// between the two centers, and the radius is the larger of |g| + r. The hull with an empty
    /// capsule is the other one (CORA returns nothing there).
    Capsule convex_hull(const Capsule &S) const;

    /// CORA's enclose: for one capsule the convex hull. CORA's three-argument form returns the
    /// hull of C and M * Cplus and never adds Cplus, so it does not enclose M * C + Cplus as
    /// documented; with (M, Cplus) it is Error::not_supported until plus is ported.
    Capsule enclose(const Capsule &S) const;
    Capsule enclose(const Tensor &M, const Capsule &Cplus) const;

    // Sums and conversions ------------------------------------------------------------------------

    /// The Minkowski sum with a capsule (CORA's plus): the centers add, the longer generator stays
    /// and the length of the shorter one joins the radius; with a vector the center shifts; with a
    /// box or a zonotope the center shifts by that of its box and the radius grows by the box's.
    Capsule plus(const Capsule &S) const;
    Capsule plus(const Tensor &v) const;
    Capsule plus(const Interval &I) const;
    Capsule plus(const Zonotope &Z) const;

    /// A zonotope that contains the capsule (CORA's zonotope): the segment plus the box around the
    /// ball, with n axis generators where CORA uses 5 n - 1 (docs/cora-differences.md).
    Zonotope zonotope() const;

    /// The ellipsoid of the capsule without a generator (CORA's ellipsoid, method "exact"); a
    /// generator needs the union of two ellipsoids, Error::no_exact_alg ("exact") or
    /// Error::not_supported ("outer", "inner"). The empty capsule gives the empty ellipsoid.
    Ellipsoid ellipsoid(const std::string &method = "exact") const;

    /// The polytope of the capsule (CORA's polytope): exact for a segment or a point, in one
    /// dimension, else method "outer" encloses it by the box in the basis of the generator
    /// ("exact": Error::no_exact_alg, "inner": Error::not_supported).
    Polytope polytope(const std::string &method = "exact") const;

    /// M * C for a matrix M (..., m, n) (CORA's mtimes): center and generator map by M, the
    /// radius scales by the Frobenius norm of M, an upper bound of its operator norm.
    Capsule linear_map(const Tensor &M) const;

    /// The capsule with its coordinates permuted: coordinate i of this one becomes coordinate
    /// dims[i] (CORA's lift_), N = dim() and dims a permutation (a capsule is bounded, so new
    /// dimensions cannot be added).
    Capsule lift(int64_t N, const std::vector<int64_t> &dims) const;

  protected:
    int64_t dim_impl() const override { return c_.shape().back(); }
    Tensor center_impl() const override;
    Set reduce_impl(int order) const override;
    Set translate_impl(const Tensor &v) const override;
    Tensor support_function_impl(const Tensor &d) const override;
    Interval interval_impl() const override;
    Tensor rand_point_impl(int64_t N, Rng &rng) const override;
    Set project_impl(const std::vector<int64_t> &dims) const override;
    std::vector<Polygon> vertices_impl() const override;
    std::vector<std::vector<Point3>> vertices_3d_impl() const override;
    std::string display_impl() const override;

    bool is_bounded_impl() const override;
    bool is_full_dim_impl(Tol tol) const override;
    Set linear_map_impl(const Tensor &M) const override;
    Set lift_impl(int64_t N, const std::vector<int64_t> &proj) const override;
    Set enclose_impl(const ContSet &S2) const override;

  private:
    /// The parts as they are, without the checks of the constructor.
    struct Unchecked {};
    /// The capsule of CORA's columns c and g (..., n, 1) and radius r (..., 1, 1), stored as the
    /// vectors (..., n) and the radius (...), with the empty members `empty`.
    Capsule(Unchecked, const Tensor &c, const Tensor &g, const Tensor &r, EmptyMask empty = {});

    /// The center and the generator as CORA's columns (..., n, 1) and the radius as (..., 1, 1),
    /// on which the algorithms run.
    Tensor cc() const { return c_.expand_dims(-1); }
    Tensor gc() const { return g_.expand_dims(-1); }
    Tensor rc() const { return r_.expand_dims(-1).expand_dims(-1); }
    Tensor rand_point_impl(int64_t N, Rng &rng, const std::string &type) const;

    Tensor c_, g_, r_;
};

/// M * C, as CORA writes it.
inline Capsule operator*(const Tensor &M, const Capsule &C) { return C.linear_map(M); }

/// C + S sums, and a point v translates, as CORA's plus; C - v translates by -v.
inline Capsule operator+(const Capsule &C, const Capsule &S) { return C.plus(S); }
inline Capsule operator+(const Capsule &C, const Interval &I) { return C.plus(I); }
inline Capsule operator+(const Capsule &C, const Zonotope &Z) { return C.plus(Z); }
inline Capsule operator+(const Capsule &C, const Tensor &v) { return C.plus(v); }
inline Capsule operator+(const Tensor &v, const Capsule &C) { return C.plus(v); }
inline Capsule operator-(const Capsule &C, const Tensor &v) { return C.plus(v * -1.0); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
