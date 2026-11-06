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
// Operations: constructors (polytope.cpp), empty, inf, enclose_points (static), constraints, box,
//             interval, center, compact, con_zonotope, contains, cart_prod, intersect,
//             convex_hull, enclose, hausdorff_distance, display (one file each). CORA's
//             conPolyZono and ellipsoid wait for ConPolyZono and Ellipsoid; the other operations
//             of the ContSet interface (support_function, rand_point, project, vertices) are
//             Error::not_supported until they are ported.
// See also:   cora/cont_set/cont_set.h, cora/cont_set/con_zonotope/con_zonotope.h
//
// CORA: contSet/@polytope/polytope.m
// CORA: contSet/@interval/polytope.m
//
// CORA authors: Viktor Kotsev, Mark Wetzlinger, Tobias Ladner, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"

#include <optional>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

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

    /// The empty set of n dimensions (CORA's polytope.empty): the row 0 <= -1 (none for n = 0)
    /// and no vertices.
    static Polytope empty(int64_t n = 0);

    /// R^n (CORA's polytope.Inf): no constraints. CORA also stores the 2^n vertices at infinity
    /// for n <= 8; vertices are not ported, so none are held here.
    static Polytope inf(int64_t n = 0);

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

    /// CORA's constraints: the held halfspaces, or those computed from the vertices: one vertex
    /// {Ae = I, be = V}; in 1D the two bounds (or the vertex as an equality); else the affine hull
    /// as equalities and the facets of the hull inside it as unit-normal inequalities, found by
    /// combinations of vertices (CORA: qhull, whose triangulated facets repeat a normal; here each
    /// facet is listed once). Not a batch.
    Constraints constraints() const;

    // Conversions and queries ---------------------------------------------------------------------

    /// The smallest box around the polytope as a polytope of its 2n axis-aligned rows (CORA's
    /// box; +-Inf bounds give no row); R^n for R^n, the empty set for the empty set. The vertex
    /// form takes the minimum and maximum of each coordinate, the halfspace form solves 2n linear
    /// programs (not a batch).
    Polytope box() const;

    /// The center: "chebyshev" (default), the center of the largest ball, from the halfspaces by a
    /// linear program; or "avg", the mean of the vertices (of the extreme ones). R^n gives the
    /// origin; the empty set a column without entries (n, 0); an unbounded set whose center is
    /// not defined gives NaN. Not a batch.
    using ContSet::center;
    Tensor center(const std::string &method) const;

    /// The set without redundancy (CORA's compact_), with the tolerance tol (default: the
    /// repository tolerance) for "is zero" and "is aligned". method: "all" (default) a minimal
    /// halfspace form; "A" the inequalities; "Ae" the equalities; "aligned" aligned rows only;
    /// "zeros" rows 0 <= b and 0 = b only; "AtoAe" pairs a x <= c, -a x <= -c as equalities;
    /// "V" the vertices that are not extreme (needs the vertex form). All but "V" normalize the
    /// rows to unit length; an empty result is empty(n), no rows left is inf(n). Not a batch.
    Polytope compact(const std::string &method = "all", Tol tol = std::nullopt) const;

    /// The constrained zonotope of the set (CORA's conZonotope, after Scott et al.):
    /// "exact:supportFunc" (default) around the box (given, else interval()), "exact:vertices"
    /// around the vertices' box. An unbounded set is Error::special_error; the empty set gives
    /// ConZonotope::empty. Not a batch.
    ConZonotope con_zonotope(const std::string &method = "exact:supportFunc",
                             const std::optional<Interval> &box = std::nullopt) const;

    /// Whether the set contains the points (n, N), all of them, or a set (CORA's contains_):
    /// "exact" (default), "exact:venum" (through the vertex form) or "exact:polymax" (through the
    /// halfspace form); "approx" is "exact" for a box or a zonotope. A point must satisfy every row
    /// up to tol (absolute); a set is contained if its support function along each row, equalities
    /// both ways, is at most the offset up to tol (within_tol). A polytope in vertex form is its
    /// points. Not a batch. tol default: the repository tolerance (docs/architecture.md). CORA's
    /// certificate and scaling outputs are not returned.
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
    /// polytope without the vertices that are not extreme. The empty set gives the empty set
    /// (CORA's; see docs/cora-differences.md), R^n gives R^n. A polytope in halfspace form is
    /// converted by enumerating its vertices. Not a batch.
    Polytope convex_hull(const Polytope &P2) const;
    Polytope convex_hull(const Interval &I) const;
    Polytope convex_hull(const Tensor &p) const;

    /// The convex hull of this set and P2 (CORA's enclose).
    Polytope enclose(const Polytope &P2) const;

    /// CORA's enclose(P, M, Pplus), the hull of P and M P + Pplus, needs the image and the sum of
    /// polytopes: Error::not_supported until they are ported.
    Polytope enclose(const Tensor &M, const Polytope &Pplus) const;

    /// The Hausdorff distance to a polytope, a box or the points (n, N) (CORA's hausdorffDist):
    /// the largest Euclidean distance of a vertex of one set to the other, found by a quadratic
    /// program per vertex. +Inf if exactly one is empty or unbounded; 0 for two equal unbounded
    /// sets, and CORA's empty-to-empty 0. Not a batch.
    double hausdorff_distance(const Polytope &P2) const;
    double hausdorff_distance(const Interval &I) const;
    double hausdorff_distance(const Tensor &points) const;

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

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
