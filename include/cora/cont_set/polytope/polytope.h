// polytope - the class of polytopes, as CORA's polytope: {x | A x <= b, Ae x = be} or conv(V)
//
// A polytope is given by its halfspaces (A, b, Ae, be) or by its vertices V, never both by the
// user; some operations add the other form (empty has both). A polytope without constraints is
// R^n. Shapes: A (..., m, n), b (..., m, 1), Ae (..., k, n), be (..., k, 1), V (..., n, p), the
// leading axes a batch. Immutable: every operation returns a new polytope. The operations that
// solve linear programs or branch on the data run on one polytope, not a batch
// (Error::not_supported). CORA stores what it has found out (emptiness, boundedness) in the object;
// here a polytope holds its data and every property is computed when it is asked for.
//
// Syntax:     Polytope P(A, b);   Polytope P(A, b, Ae, be);   Polytope P(V);   Polytope P(box);
// Operations: one file each in src/cont_set/polytope/, named as CORA's (rand_point, project,
//             vertices and the other ContSet operations as the _impl forms); the constructors
//             check their input (polytope.cpp). CORA's conPolyZono, polyZonotope, levelSet,
//             spectraShadow and matPolytope wait for their classes.
// See also:   cora/cont_set/cont_set.h, cora/cont_set/con_zonotope/con_zonotope.h
//
// CORA: contSet/@polytope/polytope.m
// CORA: contSet/@interval/polytope.m
// CORA: contSet/@emptySet/polytope.m
// CORA: contSet/@fullspace/polytope.m
// CORA authors: Viktor Kotsev, Mark Wetzlinger, Tobias Ladner, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"
#include "cora/cont_set/zono_bundle/zono_bundle.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"

#include <optional>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

class EmptySet;
class Fullspace;

class Polytope : public ContSet {
  public:
    static constexpr ClassId kClassId = ClassId::polytope;

    /// The convex hull of the vertices V (..., n, p), one per column; none (p = 0) is the empty
    /// set. The vertices must not be NaN, and finite unless n = 1. In one dimension only the
    /// smallest and the largest are kept (CORA: one if they agree).
    explicit Polytope(Tensor V);

    /// {x | A x <= b}, and with the second form also {x | Ae x = be}: A (..., m, n), b (..., m, 1),
    /// Ae (..., k, n), be (..., k, 1). A row with the offset +Inf holds everywhere and is dropped
    /// (not in a batch); an offset -Inf (or an equality offset +-Inf) makes the set empty.
    Polytope(const Tensor &A, Tensor b);
    Polytope(Tensor A, Tensor b, Tensor Ae, Tensor be);

    /// A box as a polytope: the rows x <= sup and -x <= -inf of the bounded sides (CORA's
    /// interval/polytope.m). A box of matrix bounds or without a point is refused.
    explicit Polytope(const Interval &I);

    /// The empty set or the whole space as a polytope (CORA's emptySet/polytope.m and
    /// fullspace/polytope.m): the same as empty(n) and inf(n).
    explicit Polytope(const EmptySet &E);
    explicit Polytope(const Fullspace &F);

    /// The empty set of n dimensions (CORA's polytope.empty): the row 0 <= -1 (none for n = 0)
    /// and no vertices.
    static Polytope empty(int64_t n = 0);

    /// R^n (CORA's polytope.Inf): no constraints. CORA also stores the 2^n vertices at infinity
    /// for n <= 8; vertices are not ported, so none are held here.
    static Polytope inf(int64_t n = 0);

    /// The origin {0} in n >= 1 dimensions (CORA's polytope.origin): one vertex, no halfspaces
    /// (CORA also stores the simplex x >= 0, sum x <= 0 that holds only the origin).
    static Polytope origin(int64_t n);

    /// A random polytope of dimension n (default 1..10) with CORA's generateRandom options:
    /// `nr_constraints` (default about 2.5 n), `is_degenerate` (default false: no interior) and
    /// `is_bounded` (default true). Inconsistent options are refused as in CORA.
    static Polytope generate_random(Rng &rng, std::optional<int64_t> dimension = std::nullopt,
                                    std::optional<int64_t> nr_constraints = std::nullopt,
                                    std::optional<bool> is_degenerate = std::nullopt,
                                    std::optional<bool> is_bounded = std::nullopt);

    /// The convex hull of the points (n, N), as a vertex polytope of the extreme ones in their
    /// order (CORA's enclosePoints: rank, QR and convhulln; a degenerate cloud is reduced to its
    /// affine hull first). Not a batch.
    static Polytope enclose_points(const Tensor &points);

    // Data ----------------------------------------------------------------------------------------

    /// Whether the halfspace form (A, b, Ae, be) or the vertex form V is held.
    bool is_h_rep() const noexcept { return A_.defined(); }
    bool is_v_rep() const noexcept { return V_.defined(); }

    /// The halfspace form; Error::special_error if only the vertices are held (use constraints).
    const Tensor &A() const;
    const Tensor &b() const;
    const Tensor &Ae() const;
    const Tensor &be() const;

    /// The vertices; Error::special_error if only the halfspaces are held.
    const Tensor &V() const;

    /// The halfspace form of a polytope, computed from the vertices if need be.
    struct Constraints {
        Tensor A, b, Ae, be;
    };

    /// CORA's constraints: the held halfspaces, or those computed from the vertices (the affine
    /// hull as equalities, the facets of the hull as unit-normal inequalities). Not a batch.
    Constraints constraints() const;

    // Conversions and queries ---------------------------------------------------------------------

    /// The smallest box around the polytope as a polytope of its 2n axis-aligned rows (CORA's
    /// box); R^n for R^n, the empty set for the empty set.
    Polytope box() const;

    /// The center: "chebyshev" (default), the center of the largest ball, or "avg", the mean of
    /// the extreme vertices. R^n gives the origin, the empty set (n, 0). Not a batch.
    using ContSet::center;
    Tensor center(const std::string &method) const;

    /// The set without redundancy (CORA's compact_), with the tolerance tol for "is zero" and "is
    /// aligned"; method "all" (default), "A", "Ae", "aligned", "zeros", "AtoAe" or "V". Not a
    /// batch.
    Polytope compact(const std::string &method = "all", Tol tol = std::nullopt) const;

    /// The constrained zonotope of the set (CORA's conZonotope, after Scott et al.):
    /// "exact:supportFunc" (default) or "exact:vertices". An unbounded set is
    /// Error::special_error.
    ConZonotope con_zonotope(const std::string &method = "exact:supportFunc",
                             const std::optional<Interval> &box = std::nullopt) const;

    /// Whether the set contains the points (n, N), all of them, or a set (CORA's contains_):
    /// "exact" (default), "exact:venum" or "exact:polymax", up to tol. Not a batch.
    bool contains(const Tensor &p, const std::string &method = "exact",
                  Tol tol = std::nullopt) const;
    bool contains(const ContSet &S, const std::string &method = "exact",
                  Tol tol = std::nullopt) const;

    // Combining with other sets -------------------------------------------------------------------

    /// The intersection with a polytope, a box or a point (n, 1) (CORA's and_): the rows of both
    /// stack, exactly, as halfspaces; a point gives itself if it is contained, else the empty set.
    /// R^n is neutral. A polytope in vertex form is converted by constraints (not a batch).
    Polytope intersect(const Polytope &S) const;
    Polytope intersect(const Interval &I) const;
    Polytope intersect(const Tensor &p, Tol tol = std::nullopt) const;

    /// The Cartesian product with a polytope, a box or a point (n2, 1), exact: halfspaces on the
    /// block diagonal, or all pairs of vertices if both are in vertex form. R^n times a point is
    /// not R^(n+n2): the point's equalities are rows. A mixed pair converts the vertex form.
    Polytope cart_prod(const Polytope &P2) const;
    Polytope cart_prod(const Interval &I) const;
    Polytope cart_prod(const Tensor &p) const;

    /// The Cartesian product of a point p (n1, 1) with a polytope.
    static Polytope cart_prod(const Tensor &p, const Polytope &P);

    /// The convex hull with a polytope, a box or a point (n, 1) (CORA's convHull_), as a vertex
    /// polytope of the extreme vertices. The hull with R^n is R^n. Not a batch.
    Polytope convex_hull(const Polytope &P2) const;
    Polytope convex_hull(const Interval &I) const;
    Polytope convex_hull(const Tensor &p) const;

    /// The convex hull of this set and P2 (CORA's enclose).
    Polytope enclose(const Polytope &P2) const;

    /// CORA's enclose(P, M, Pplus), the hull of P and M P + Pplus, needs the image and the sum of
    /// polytopes: Error::not_supported until they are ported.
    Polytope enclose(const Tensor &M, const Polytope &Pplus) const;

    /// The Hausdorff distance to a polytope, a box or the points (n, N) (CORA's hausdorffDist):
    /// the largest distance of a vertex of one set to the other, by quadratic programs. Not a
    /// batch.
    double hausdorff_distance(const Polytope &P2) const;
    double hausdorff_distance(const Interval &I) const;
    double hausdorff_distance(const Tensor &points) const;

    // Queries -------------------------------------------------------------------------------------

    /// Whether the set is also a set of the CORA type ("origin", "point", "interval", "hyperplane",
    /// "halfspace", "emptySet", "fullspace", "conZonotope", ...); the converted set is not
    /// returned. "zonotope" and "parallelotope" are Error::not_supported, as in CORA. Not a batch.
    bool representsa(const std::string &type, Tol tol = std::nullopt) const;

    /// The volume: Inf if unbounded, 0 if empty or flat (CORA's volume_, by the hull of the
    /// vertices). Not a batch.
    double volume() const;

    /// Whether the set has a nonempty interior, and a basis (n, k) of the affine subspace that
    /// holds it (the identity if full-dimensional); CORA's isFullDim with two outputs. Not a batch.
    bool is_full_dim(Tol tol = std::nullopt) const;
    std::pair<bool, Tensor> is_full_dim_subspace(Tol tol = std::nullopt) const;

    /// Whether the set equals a polytope, a point (n, 1) or another set (CORA's isequal): by the
    /// rows, by the vertices or by containment both ways. tol default: the repository's. Not a
    /// batch.
    bool isequal(const Polytope &S, Tol tol = std::nullopt) const;
    bool isequal(const ContSet &S, Tol tol = std::nullopt) const;
    bool isequal(const Tensor &p, Tol tol = std::nullopt) const;

    /// Whether the set meets a polytope, a box, a zonotope, a constrained zonotope, a bundle or
    /// points (n, N) (CORA's isIntersecting_): type "exact" (default; a linear program) or "approx"
    /// (support functions along the rows; not for a polytope). Not a batch.
    bool is_intersecting(const Polytope &S, const std::string &type = "exact",
                         Tol tol = std::nullopt) const;
    bool is_intersecting(const Interval &I, const std::string &type = "exact",
                         Tol tol = std::nullopt) const;
    bool is_intersecting(const Zonotope &Z, const std::string &type = "exact",
                         Tol tol = std::nullopt) const;
    bool is_intersecting(const ConZonotope &S, const std::string &type = "exact",
                         Tol tol = std::nullopt) const;
    bool is_intersecting(const ZonoBundle &S, const std::string &type = "exact",
                         Tol tol = std::nullopt) const;
    bool is_intersecting(const Tensor &p, const std::string &type = "exact",
                         Tol tol = std::nullopt) const;

    /// Whether this set is contained in S (CORA's le, P <= S): S contains the set, up to tol.
    bool less_equal(const ContSet &S, Tol tol = std::nullopt) const;

    /// The Euclidean distance to a polytope, or of each point (n, N) to the set, shape (1, N):
    /// Inf for the empty set, 0 for a set that meets (CORA's distance). Not a batch.
    double distance(const Polytope &S) const;
    Tensor distance(const Tensor &points) const;

    // Maps and sums -------------------------------------------------------------------------------

    /// The Minkowski sum with a polytope, a box or a column v (n, 1) (CORA's plus): exact; R^n
    /// absorbs, the empty set annihilates, the sum of two halfspace sets is the projection of their
    /// block product, of two vertex sets all pairs of vertices.
    Polytope plus(const Polytope &S) const;
    Polytope plus(const Interval &I) const;
    Polytope plus(const Tensor &v) const;

    /// The set shifted by -v (CORA's minus with a vector).
    using ContSet::minus;
    Polytope minus(const Tensor &v) const;

    /// M P for a matrix M (..., m, n) (CORA's mtimes): exact; vertices map, halfspaces by the
    /// inverse of a square M or the singular value decomposition of a wide or tall one. The zero
    /// matrix gives the origin. An interval matrix is Error::not_supported.
    Polytope linear_map(const Tensor &M) const;

    /// The set without the other coordinates (the dimensions `dims`, 0-based, in that order):
    /// vertices by selection, halfspaces by Fourier-Motzkin elimination. Not a batch for
    /// halfspaces.
    Polytope project(const std::vector<int64_t> &dims) const;

    /// The set in N dimensions with coordinate j at position dims[j] (CORA's lift_): the rows keep
    /// their columns, the others are free; the vertices of a lifted set would be unbounded, so only
    /// the halfspace form is returned. project_high_dim sets the other coordinates to zero.
    Polytope lift(int64_t N, const std::vector<int64_t> &dims) const;
    Polytope project_high_dim(int64_t N, const std::vector<int64_t> &dims) const;

    /// The convex hull of the union (CORA's or is convHull_): convex_hull.
    Polytope unite(const Polytope &S) const;
    Polytope unite(const Interval &I) const;
    Polytope unite(const Tensor &p) const;

    /// The complement (CORA's not): one inequality gives the opposite one shifted by eps, the empty
    /// set R^n and R^n the empty set; anything else is Error::not_supported.
    Polytope complement() const;

    /// The same set with the rows scaled (CORA's normalizeConstraints): type "b" (default; the
    /// offsets to +-1, equalities to 1), "A" (unit-length rows), "be" and "Ae" for the equalities
    /// only. Halfspace form only.
    Polytope normalize_constraints(const std::string &type = "b") const;

    /// One halfspace that encloses (type "outer", default) or lies inside ("inner") the set over
    /// the box dom (CORA's reduceOverDomain): the average of the normalized rows against the
    /// offsets over dom.
    Polytope reduce_over_domain(const Interval &dom, const std::string &type = "outer") const;

    /// An enclosure by 2 * order * n random supporting halfspaces (CORA's reduce with 'rand'),
    /// directions drawn with rng. Not a batch.
    Polytope reduce(int order, Rng &rng, const std::string &method = "rand") const;

    /// The Minkowski difference {x | x + S in this set} (CORA's minkDiff, through
    /// ContSet::minkowski_difference) of a set with a support function: type "exact" (default; the
    /// offsets of the rows lowered by the support function of S) or "exact:vertices".
    using ContSet::minkowski_difference;
    Polytope minkowski_difference(const Tensor &p) const;

    /// The pieces of the set difference this \ S (CORA's mldivide): polytopes whose union is the
    /// closure of the difference. Only full-dimensional operands (CORA's other branches are
    /// Error::not_supported); a corrected version of CORA's, which keeps only the last piece.
    std::vector<Polytope> set_difference(const Polytope &S) const;

    // Conversions ---------------------------------------------------------------------------------

    /// A zonotope around the bounded set (CORA's zonotope): "outer" (default) is the box of the
    /// covariance ellipsoid of the vertices (CORA's ellipsoid is a semidefinite program), "inner"
    /// and "exact" are Error::not_supported. Not a batch.
    Zonotope zonotope(const std::string &method = "outer") const;

    /// The bundle of the strips along the rows of the bounded set (CORA's zonoBundle). Not a batch.
    ZonoBundle zono_bundle() const;

    /// An ellipsoid that encloses the bounded set (CORA's ellipsoid): "outer" (default; the
    /// covariance ellipsoid of the vertices); "outer:min-vol" and "inner" are Error::not_supported
    /// (semidefinite programs). Not a batch.
    Ellipsoid ellipsoid(const std::string &method = "outer") const;

    /// N random points (n, N) of the set (CORA's randPoint_): "standard" (default; a random convex
    /// combination of extreme points), "extreme" or "uniform:hitAndRun" (full-dimensional sets);
    /// the other uniform walks are Error::not_supported. Not a batch.
    using ContSet::rand_point;
    Tensor rand_point(int64_t N, Rng &rng, const std::string &type) const;

  protected:
    int64_t dim_impl() const override;
    Tensor center_impl() const override;
    Tensor support_function_impl(const Tensor &d) const override;
    Interval interval_impl() const override;
    Tensor rand_point_impl(int64_t N, Rng &rng) const override;
    Set project_impl(const std::vector<int64_t> &dims) const override;
    std::vector<Polygon> vertices_impl() const override;
    std::vector<std::vector<Point3>> vertices_3d_impl() const override;
    std::string display_impl() const override;

    // the overrides of the queries and maps of the second batch
    bool is_bounded_impl() const override;
    bool is_full_dim_impl(Tol tol) const override;
    Set translate_impl(const Tensor &v) const override;
    Set linear_map_impl(const Tensor &M) const override;
    Set lift_impl(int64_t N, const std::vector<int64_t> &proj) const override;
    Set project_high_dim_impl(int64_t N, const std::vector<int64_t> &proj) const override;
    Set minkowski_difference_impl(const ContSet &S2,
                                  const std::optional<std::string> &type) const override;
    std::optional<std::vector<std::string>> compact_methods_impl() const override;
    Set compact_impl(const std::optional<std::string> &method, Tol tol) const override;
    Set enclose_impl(const ContSet &S2) const override;

  private:
    /// The parts of a polytope that is built by an operation (in either or both forms).
    struct Parts {
        Tensor A, b, Ae, be, V;
    };
    explicit Polytope(Parts parts);

    Tensor A_, b_, Ae_, be_, V_;
};

// Operators, as CORA writes them: M * P maps, P + S sums, P - v shifts.
inline Polytope operator*(const Tensor &M, const Polytope &P) { return P.linear_map(M); }
inline Polytope operator+(const Polytope &P, const Polytope &S) { return P.plus(S); }
inline Polytope operator+(const Polytope &P, const Tensor &v) { return P.plus(v); }
inline Polytope operator+(const Tensor &v, const Polytope &P) { return P.plus(v); }
inline Polytope operator-(const Polytope &P, const Tensor &v) { return P.minus(v); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
