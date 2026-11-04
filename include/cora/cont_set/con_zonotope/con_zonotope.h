// con_zonotope - the class of constrained zonotopes, as CORA's conZonotope
//
// The center c is a column (..., n, 1), the generators G are (..., n, m), the constraint matrix A
// is (..., p, m) and its offset b a column (..., p, 1); p = 0 is a zonotope. Immutable: every
// operation returns a new set. center() is CORA's center (a point inside the set), c() the stored
// offset. The set is empty if the constraints have no solution in the box; empty(n) is such a set.
//
// Syntax:     ConZonotope cz(c, G, A, b);   ConZonotope cz(Z);   ConZonotope::empty(n);
// Operations: constructors (con_zonotope.cpp), empty, center, interval, is_bounded, is_full_dim,
//             compact, cart_prod, intersect, con_intersect, enclose, convex_hull, cubic_map,
//             intersect_strip, contains, approximate_bounds, display, origin, plus, linear_map,
//             lift, reduce, quad_map, rescale, representsa, isequal, is_intersecting, split, unite,
//             minkowski_difference, project (one file each). The other operations of the ContSet
//             interface (support_function, rand_point, vertices) are Error::not_supported until
//             they are ported.
// See also:   cora/cont_set/cont_set.h, cora/cont_set/zonotope/zonotope.h
//
// CORA: contSet/@conZonotope/conZonotope.m

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

class ConZonotope : public ContSet {
  public:
    static constexpr ClassId kClassId = ClassId::con_zonotope;

    /// The constrained zonotope of the center c (..., n, 1), the generators G (..., n, m) and the
    /// constraints A b = b (A (..., p, m), b (..., p, 1)); without A and b no constraint.
    ConZonotope(const Tensor &c, const Tensor &G);
    ConZonotope(Tensor c, Tensor G, Tensor A, Tensor b);

    /// A zonotope or a box as a constrained zonotope without constraints (CORA's conversions).
    explicit ConZonotope(const Zonotope &Z);
    explicit ConZonotope(const Interval &I);

    /// CORA's legacy forms conZonotope(Z) and conZonotope(Z, A, b): Z = [c G] (..., n, 1 + m); no
    /// constraints if A and b are not given.
    static ConZonotope from_matrix(const Tensor &Z, const std::optional<Tensor> &A = std::nullopt,
                                   const std::optional<Tensor> &b = std::nullopt);

    /// The stored center c, generators G, constraint matrix A and offset b.
    const Tensor &c() const noexcept { return c_; }
    const Tensor &generators() const noexcept { return G_; }
    const Tensor &A() const noexcept { return A_; }
    const Tensor &b() const noexcept { return b_; }

    /// The empty set of n dimensions (CORA's conZonotope.empty): no generators and the constraint
    /// 0 = 1, so that every operation on it stays an enclosure (CORA's has a center without
    /// columns, which and_ and cartProd_ mishandle; docs/cora-differences.md).
    static ConZonotope empty(int64_t n = 0);

    // Queries -------------------------------------------------------------------------------------

    /// Always true: the generators are bounded by the unit box.
    bool is_bounded() const;

    /// Whether the set has a nonempty interior: the generators, restricted to the directions the
    /// constraints and the box leave free, span the space (singular values up to tol count as
    /// zero). An empty set has none. With the second form also a basis (n, r) of the span (the
    /// identity if full). Not a batch if there are constraints. tol default: the repository
    /// tolerance (docs/architecture.md).
    bool is_full_dim(Tol tol = std::nullopt) const;
    std::pair<bool, Tensor> is_full_dim_subspace(Tol tol = std::nullopt) const;

    /// Whether the set contains the points p (n, N) (all of them), a zonotope or a box, up to tol
    /// in the norm of the set (a linear program per point). method: "exact" (and "exact:venum", the
    /// vertices of a zonotope or box are the points); "approx", "approx:st", "exact:polymax",
    /// "sampling" and "dual" ones are Error::not_supported. Not a batch. tol default: the
    /// repository tolerance.
    bool contains(const Tensor &p, const std::string &method = "exact",
                  Tol tol = std::nullopt) const;
    bool contains(const Zonotope &S, const std::string &method = "exact",
                  Tol tol = std::nullopt) const;
    bool contains(const Interval &I, const std::string &method = "exact",
                  Tol tol = std::nullopt) const;

    // Combining with other sets -------------------------------------------------------------------

    /// Without the constraint rows 0 = 0 and the generators that are zero and appear in no
    /// constraint (method "zeros", the default, or "all"); "aligned" leaves the set as it is.
    /// Only exact zeros count, as CORA's compact_; tol is checked. A batch drops what every
    /// member drops.
    ConZonotope compact(const std::optional<std::string> &method = std::nullopt,
                        Tol tol = std::nullopt) const;

    /// The Cartesian product with a constrained zonotope, a zonotope, a box or a point (n2, 1):
    /// exact; centers stack, generators and constraints sit on the block diagonal.
    ConZonotope cart_prod(const ConZonotope &S) const;
    ConZonotope cart_prod(const Zonotope &Z) const;
    ConZonotope cart_prod(const Interval &I) const;
    ConZonotope cart_prod(const Tensor &p) const;

    /// The Cartesian product of a point p (n1, 1) with a constrained zonotope.
    static ConZonotope cart_prod(const Tensor &p, const ConZonotope &cz);

    /// The intersection with a constrained zonotope, a zonotope or a box (CORA's and_): exact; the
    /// factors of both stay and one equality per dimension makes the points agree. Without zero
    /// rows and generators (compact).
    ConZonotope intersect(const ConZonotope &S) const;
    ConZonotope intersect(const Zonotope &Z) const;
    ConZonotope intersect(const Interval &I) const;

    /// The points of this set whose image under M (n2, n) is in cz2 (CORA's conIntersect): exact.
    ConZonotope con_intersect(const ConZonotope &cz2, const Tensor &M) const;

    /// A constrained zonotope that encloses this one and cz2 (CORA's enclose); cz2 must carry the
    /// constraints of this set (it is a linear image of it), else Error::special_error. With
    /// (M, S) the second set is M * this + S for a constrained zonotope S, or + t for a point t.
    ConZonotope enclose(const ConZonotope &cz2) const;
    ConZonotope enclose(const Tensor &M, const ConZonotope &S) const;
    ConZonotope enclose(const Tensor &M, const Tensor &t) const;

    /// The convex hull with a constrained zonotope, a zonotope, a box or a point (n, 1) (CORA's
    /// convHull_): exact. method "exact" and "exact:Raghuraman" (default) or "exact:Kochdumper".
    ConZonotope convex_hull(const ConZonotope &S,
                            const std::string &method = "exact:Raghuraman") const;
    ConZonotope convex_hull(const Zonotope &Z,
                            const std::string &method = "exact:Raghuraman") const;
    ConZonotope convex_hull(const Interval &I,
                            const std::string &method = "exact:Raghuraman") const;
    ConZonotope convex_hull(const Tensor &p, const std::string &method = "exact:Raghuraman") const;

    /// The convex hull of this set and the others at once (CORA's convHull_ with a list).
    ConZonotope convex_hull(const std::vector<ConZonotope> &others) const;

    /// An enclosure of {(x' T_i x) x | x in cz}: T[i][k] are the (n, n) matrices of output i,
    /// z_i = sum_k x_k (x' T[i][k] x); ind[i] lists the k that enter output i (empty: all). The
    /// factors of the linear terms keep the constraints; every product term is a new free factor.
    /// Not a batch if there are constraints.
    ConZonotope cubic_map(const std::vector<std::vector<Tensor>> &T,
                          const std::vector<std::vector<int64_t>> &ind = {}) const;

    /// An enclosure of {(x1' T_i x2) x3 | x1 in cz, x2 in cz2, x3 in cz3}, with z_i as above.
    ConZonotope cubic_map(const ConZonotope &cz2, const ConZonotope &cz3,
                          const std::vector<std::vector<Tensor>> &T,
                          const std::vector<std::vector<int64_t>> &ind = {}) const;

    /// The matrices E, F, A, C of an observer, for the gain "wang-FRad" of intersect_strip.
    using StripObserver = Zonotope::StripObserver;

    /// The intersection with the strips |C x - y| <= phi (C (k, n), phi and y (k, 1)), exact for a
    /// gain L (n, k): center c + L (y - C c), generators [(I - L C) G, L diag(phi)], and one
    /// equality per strip. The gain is given, or chosen by "normGen" (default), "alamo-FRad",
    /// "bravo" (an unconstrained zonotope around the one strip), or built from an observer
    /// ("wang-FRad"); "svd" and "radius" (an optimizer) are not supported.
    ConZonotope intersect_strip(const Tensor &C, const Tensor &phi, const Tensor &y,
                                const std::string &method = "normGen") const;
    ConZonotope intersect_strip(const Tensor &C, const Tensor &phi, const Tensor &y,
                                const Tensor &lambda) const;
    ConZonotope intersect_strip(const Tensor &C, const Tensor &phi, const Tensor &y,
                                const StripObserver &aux) const;

    // Maps and sums -------------------------------------------------------------------------------

    /// The origin {0} in n dimensions (no generators, no constraints), n >= 1.
    static ConZonotope origin(int64_t n);

    /// The Minkowski sum with a constrained zonotope, a zonotope, a box or a column v (n, 1)
    /// (CORA's plus): exact; centers add, generators join, constraints on the block diagonal.
    ConZonotope plus(const ConZonotope &S) const;
    ConZonotope plus(const Zonotope &Z) const;
    ConZonotope plus(const Interval &I) const;
    ConZonotope plus(const Tensor &v) const;

    /// M * cz for a matrix M (..., m, n) (CORA's mtimes): exact; the center and generators map, the
    /// constraints stay.
    ConZonotope linear_map(const Tensor &M) const;

    /// The set in N dimensions whose state dims[i] is state i (a permutation: new dimensions would
    /// be unbounded). CORA's lift_ projects onto dims, the inverse permutation
    /// (docs/cora-differences.md).
    ConZonotope lift(int64_t N, const std::vector<int64_t> &dims) const;

    // Reductions and queries ----------------------------------------------------------------------

    /// A set that encloses this one with at most about order * n generators (CORA's reduce with
    /// "girard"): Girard's rule on the zonotope [G; A] that the constraints lift the set to; the
    /// constraints stay. Without constraints a zonotope's reduce. Not a batch for the rule's
    /// ranking is per member. Other methods of CORA are Error::not_supported.
    ConZonotope reduce(int order = 1, const std::string &method = "girard") const;

    /// An enclosure of {(x' Q_i x)_i | x in cz}: Q[i] are the (n, n) matrices of the outputs. The
    /// linear terms keep the constraints, the squares and the mixed terms are new free factors; a
    /// constrained set is rescaled first ("iter").
    ConZonotope quad_map(const std::vector<Tensor> &Q) const;

    /// An enclosure of {(x1' Q_i x2)_i | x1 in cz, x2 in cz2}.
    ConZonotope quad_map(const ConZonotope &cz2, const std::vector<Tensor> &Q) const;

    /// The same set with every factor scaled to the range it takes (CORA's rescale): method
    /// "exact" (a linear program per bound) or "iter" (bounds from the constraints by interval
    /// arithmetic, after a Gauss-Jordan elimination of A). An empty set is Error::empty_set, a set
    /// without constraints Error::special_error, as in CORA. Not a batch.
    ConZonotope rescale(const std::string &method = "exact") const;

    /// Whether the set is also a set of the CORA type ("origin", "point", "interval", "zonotope",
    /// "emptySet", "polytope", "hyperplane", ...); the converted set is not returned. "emptySet"
    /// is a linear program. "conHyperplane", "capsule", "ellipsoid", "levelSet" and
    /// "parallelotope" are Error::not_supported, as in CORA. Not a batch. tol default: the
    /// repository tolerance.
    bool representsa(const std::string &type, Tol tol = std::nullopt) const;

    /// Whether the set equals a constrained zonotope, a zonotope or a box (CORA's isequal): the
    /// same data up to the order of the factors, else the vertices of each lie in the other. Not
    /// a batch. tol default: the repository tolerance (CORA's own is 1e-12).
    bool isequal(const ConZonotope &S, Tol tol = std::nullopt) const;
    bool isequal(const Zonotope &Z, Tol tol = std::nullopt) const;
    bool isequal(const Interval &I, Tol tol = std::nullopt) const;

    /// Whether the set and a constrained zonotope, a zonotope, a box or points (n, N) (all of
    /// them) intersect. type "exact": the intersection is not empty (a linear program); "approx":
    /// the boxes around both overlap. Not a batch. tol: for points, the tolerance of contains;
    /// default: the repository tolerance.
    bool is_intersecting(const ConZonotope &S, const std::string &type = "exact",
                         Tol tol = std::nullopt) const;
    bool is_intersecting(const Zonotope &Z, const std::string &type = "exact",
                         Tol tol = std::nullopt) const;
    bool is_intersecting(const Interval &I, const std::string &type = "exact",
                         Tol tol = std::nullopt) const;
    bool is_intersecting(const Tensor &p, const std::string &type = "exact",
                         Tol tol = std::nullopt) const;

    /// The two halves of the set cut at the center of coordinate N (0-based) of its box (CORA's
    /// split(cZ, N)): the intersections with the halves of the box; and the list for every
    /// coordinate (split(cZ)). They cover the set. Not a batch.
    std::pair<ConZonotope, ConZonotope> split(int64_t N) const;
    std::vector<std::pair<ConZonotope, ConZonotope>> split() const;

    /// A constrained zonotope without constraints that encloses the union with the sets (CORA's
    /// or, "linprog"): the iterative union of the sets' zonotopes, its generators scaled and its
    /// center moved to the least that cover every set along its facets. "tedrake" (which fails in
    /// CORA) and an order reduction are Error::not_supported. Not a batch.
    ConZonotope unite(const std::vector<ConZonotope> &others, const std::string &method = "linprog",
                      std::optional<int> order = std::nullopt) const;
    ConZonotope unite(const ConZonotope &S, const std::string &method = "linprog",
                      std::optional<int> order = std::nullopt) const;
    ConZonotope unite(const Zonotope &Z, const std::string &method = "linprog",
                      std::optional<int> order = std::nullopt) const;
    ConZonotope unite(const Interval &I, const std::string &method = "linprog",
                      std::optional<int> order = std::nullopt) const;
    ConZonotope unite(const Tensor &p, const std::string &method = "linprog",
                      std::optional<int> order = std::nullopt) const;

    /// The Minkowski difference {x | x + S in this set} (CORA's minkDiff) with a zonotope, a box
    /// or a point (n, 1): type "exact" (default) intersects the translates by the vertices of S,
    /// "inner:vertices" those by the center and generators of S, "inner:Vinod" scales the
    /// factors (a zonotope or box S). The empty result is empty(n). Not a batch.
    using ContSet::minkowski_difference;
    ConZonotope minkowski_difference(const Tensor &p) const;

    // Bounds of batches ---------------------------------------------------------------------------

    /// CORA's options.nn of approximateBoundsWithGPU.
    struct BoundsOptions {
        std::string method = "fourier-motzkin"; ///< or "exact" (a linear program per bound)
        int max_iter = 4;                       ///< passes of the Fourier-Motzkin tightening
        bool batch_union = true;                ///< the union constraints become batch members
    };

    /// The bounds of a batch of constrained zonotopes: the box (l, u) (..., n, 1) and the bounds
    /// (bl, bu) (..., q, 1) of the factors; NaN and 0 for an empty member.
    struct Bounds {
        Tensor l, u, bl, bu;
    };

    /// Bounds of the set c + G x + [-dr, dr], |x|_inf <= 1, A x <= b, for a batch with c (B, n, 1),
    /// G (B, n, m), A (B, p, q), b (B, p, 1), q >= m and dr (B, n, 1) (default: 0). Unlike the
    /// class, the constraints are inequalities, and the first num_union_const rows are a union
    /// (the rest an intersection). "fourier-motzkin" tightens the factor bounds row by row
    /// (Koller, ICLR 2026, Prop. 3); "dual-iter" is Error::not_supported.
    static Bounds approximate_bounds(const Tensor &c, const Tensor &G, const Tensor &A,
                                     const Tensor &b,
                                     const std::optional<Tensor> &dr = std::nullopt,
                                     int num_union_const = 1,
                                     const std::optional<BoundsOptions> &options = std::nullopt);

  protected:
    int64_t dim_impl() const override { return c_.shape()[c_.shape().size() - 2]; }
    Tensor center_impl() const override;
    Tensor support_function_impl(const Tensor &d) const override;
    Interval interval_impl() const override;
    Tensor rand_point_impl(int64_t N, Rng &rng) const override;
    Set project_impl(const std::vector<int64_t> &dims) const override;
    std::vector<Polygon> vertices_impl() const override;
    std::vector<std::vector<Point3>> vertices_3d_impl() const override;
    std::string display_impl() const override;

    // The generic operations of ContSet, which a constrained zonotope answers by its own.
    Set translate_impl(const Tensor &v) const override;
    Set uminus_impl() const override;
    Set linear_map_impl(const Tensor &M) const override;
    Set reduce_impl(int order) const override;
    Set lift_impl(int64_t N, const std::vector<int64_t> &proj) const override;
    Set minkowski_difference_impl(const ContSet &S2,
                                  const std::optional<std::string> &type) const override;
    bool is_bounded_impl() const override;
    bool is_full_dim_impl(Tol tol) const override;

  private:
    Tensor c_, G_, A_, b_;
};

// Operators, as CORA writes them: M * cz is the linear map, s * cz scales, cz + S is the Minkowski
// sum, a column v translates.
inline ConZonotope operator*(const Tensor &M, const ConZonotope &cz) { return cz.linear_map(M); }
inline ConZonotope operator+(const ConZonotope &cz, const ConZonotope &S) { return cz.plus(S); }
inline ConZonotope operator+(const ConZonotope &cz, const Tensor &v) { return cz.plus(v); }
inline ConZonotope operator+(const Tensor &v, const ConZonotope &cz) { return cz.plus(v); }
inline ConZonotope operator-(const ConZonotope &cz, const Tensor &v) { return cz.plus(v * -1.0); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
