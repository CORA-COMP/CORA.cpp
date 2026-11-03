// zonotope - the class of zonotopes Z = {c + G b | |b|_inf <= 1}, as CORA's zonotope
//
// The center c is a column (..., n, 1); the generators G are (..., n, m), one per column.
// Immutable: every operation returns a new zonotope.
//
// Syntax:     Zonotope Z(c, G);   Zonotope Z = Zonotope::generate_random(n, m, rng);
// Operators:  M * Z (linear_map), s * Z, Z + Z2 and Z + I (plus), Z + v, Z - v, -Z, as in CORA
// Operations: linear_map, plus, linear_combination, support_function, interval, rand_point,
//             generate_random, stack, project, vertices, reduce, display, abs, box, center,
//             compact, cart_prod, enclose, convex_hull, intersect, constraints_satisfied,
//             cubic_map, hausdorff_distance_to_box, dominant_directions, contains,
//             boundary_point, exact_plus, generator_length, is_bounded, is_full_dim, isequal,
//             is_intersecting, lift, norm, minnorm, intersect_strip, enclose_points,
//             enclose_many, mink_diff (one file each, as in CORA's @zonotope); the constructor
//             checks its input (zonotope.cpp)
// See also:   cora/cont_set/cont_set.h, cora/cont_set/interval/interval.h
//
// CORA: contSet/@zonotope/zonotope.m
//
// Not the competition's zonotope (competition/sets): that one is tuned for speed on one library.

#pragma once

#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"

#include <limits>
#include <optional>
#include <string>
#include <utility>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

class Zonotope : public ContSet {
  public:
    static constexpr ClassId kClassId = ClassId::zonotope;

    /// The zonotope of the center c (..., n, 1) and the generators G (..., n, m).
    Zonotope(Tensor c, Tensor G);

    /// A random zonotope with m generators in n dimensions (CORA: center 10*randn, generators of
    /// uniformly random direction and length ~ U[0, 1]).
    static Zonotope generate_random(int64_t n, int64_t m, Rng &rng);

    // Construction --------------------------------------------------------------------------------

    /// The center c and the generators G.
    const Tensor &center() const noexcept { return c_; }
    const Tensor &generators() const noexcept { return G_; }

    using ContSet::rand_point;

    /// N random points c + G b, columns (..., n, N): b ~ U[-1, 1]^m for type "standard", b from
    /// the corners {-1, 1}^m (points on the boundary) for "extreme".
    Tensor rand_point(int64_t N, Rng &rng, const std::string &type) const;

    // Linear and set operations -------------------------------------------------------------------

    /// M * Z for a matrix M (..., k, n): exact, c -> Mc and G -> MG.
    Zonotope linear_map(const Tensor &M) const;

    /// [M] * Z for an interval matrix (an Interval with (..., n, n) bounds): an enclosure.
    Zonotope linear_map(const Interval &M) const;

    /// The Minkowski sum: the centers add, the generators are joined.
    Zonotope plus(const Zonotope &Z2) const;

    /// The Minkowski sum with a box: its center adds, its radius becomes n axis generators.
    Zonotope plus(const Interval &I) const;

    /// A batch of zonotopes of one dimension (and batch shape) as one; those with fewer generators
    /// than the most are padded with zero generators.
    static Zonotope stack(const std::vector<Zonotope> &Zs);

    /// A zonotope of at most order * n generators (n the dimension) that contains this one, by
    /// Girard's method: the longest generators stay, the others become a box. Not a batch.
    Zonotope reduce(int order) const;

    /// Encloses the segments between matching points of Z and Z2 = M*Z + t (same generator
    /// factors): 2m + 1 generators.
    Zonotope linear_combination(const Zonotope &Z2) const;

    /// |c| and |G| entrywise, as CORA computes it: not the image of the set under abs.
    Zonotope abs() const;

    /// The box around the zonotope as a zonotope: the same center, one axis generator per
    /// dimension with the radius of that dimension.
    Zonotope box() const;

    /// Without the generators of norm <= tol (method "zeros"); "all" also merges the generators
    /// that are parallel up to tol, "aligned" leaves the set as it is (as CORA). A batch drops
    /// what every member drops; "all" is not for a batch.
    /// tol default: the repository tolerance (docs/architecture.md).
    Zonotope compact(const std::string &method = "zeros", Tol tol = std::nullopt) const;

    // Combining with other sets -------------------------------------------------------------------

    /// The Cartesian product with a zonotope, a box or a point (n2, 1): centers stack,
    /// generators go on the block diagonal.
    Zonotope cart_prod(const Zonotope &Z2) const;
    Zonotope cart_prod(const Interval &I) const;
    Zonotope cart_prod(const Tensor &p) const;

    /// The Cartesian product of a point p (n1, 1) with a zonotope.
    static Zonotope cart_prod(const Tensor &p, const Zonotope &Z);

    /// A zonotope that encloses both (CORA's enclose): the generators in common are combined
    /// as in linear_combination, the others of the longer one are kept.
    Zonotope enclose(const Zonotope &Z2) const;

    /// A zonotope that encloses the convex hull of the zonotope and a zonotope, a box or a point.
    Zonotope convex_hull(const Zonotope &Z2) const;
    Zonotope convex_hull(const Interval &I) const;
    Zonotope convex_hull(const Tensor &p) const;

    /// A zonotope that encloses the intersection with a zonotope. Two boxes intersect exactly (an
    /// error if they do not meet, as the empty set is not a class yet). Otherwise "averaging"
    /// weights the generators of both; "conZonotope" (CORA's default) needs ConZonotope.
    Zonotope intersect(const Zonotope &Z2, const std::string &method = "conZonotope") const;
    Zonotope intersect(const Interval &I, const std::string &method = "conZonotope") const;

    /// Whether C x < d for every x in the zonotope (sufficient: false says nothing); C (k, n)
    /// and d (k, 1). For a batch, whether it holds for every member.
    bool constraints_satisfied(const Tensor &C, const Tensor &d) const;

    /// An enclosure of {(x' T_i x) x | x in Z}: T[i][k] are the (n, n) matrices of output i,
    /// z_i = sum_k x_k (x' T[i][k] x); ind[i] lists the k that enter output i (empty: all).
    Zonotope cubic_map(const std::vector<std::vector<Tensor>> &T,
                       const std::vector<std::vector<int64_t>> &ind = {}) const;

    /// An enclosure of {(x1' T_i x2) x3 | x1 in Z, x2 in Z2, x3 in Z3}, with z_i as above.
    Zonotope cubic_map(const Zonotope &Z2, const Zonotope &Z3,
                       const std::vector<std::vector<Tensor>> &T,
                       const std::vector<std::vector<int64_t>> &ind = {}) const;

    /// A bound on the Euclidean Hausdorff distance between the zonotope and its box, (..., 1, 1):
    /// method "exact" (sampled directions, CORA's default for n <= 3), "naive" (the norm of the
    /// radius, the default above), "wgreedy" or "wopt". "ell" is not supported.
    Tensor hausdorff_distance_to_box(const std::optional<std::string> &method = std::nullopt) const;

    /// n generators (as columns, (n, n)) that span a parallelotope of small volume around the
    /// zonotope: of the filter_length1 longest generators, the n-subsets of the filter_length2
    /// largest determinants, and of them the least volume (CORA: n + 5 and n + 3). Not a batch.
    Tensor dominant_directions(std::optional<int> filter_length1 = std::nullopt,
                               std::optional<int> filter_length2 = std::nullopt) const;

    /// Whether the zonotope contains the points p (n, N) (all of them), a zonotope or a box, up to
    /// tol (the norm of the zonotope, relaxed as within_tol). method: "exact" and "exact:venum"
    /// (a linear program per point; a search over the vertices for a set); "approx" and
    /// "approx:st" (a sufficient test of Sadraddini and Tedrake for a set); "exact:polymax"
    /// needs Polytope, "approx:stDual", "sampling" and "opt" are not supported. Not a batch. tol
    /// default: the repository tolerance (docs/architecture.md).
    bool contains(const Tensor &p, const std::string &method = "exact",
                  Tol tol = std::nullopt) const;
    bool contains(const Zonotope &S, const std::string &method = "exact",
                  Tol tol = std::nullopt) const;
    bool contains(const Interval &I, const std::string &method = "exact",
                  Tol tol = std::nullopt) const;

    /// The point where the ray from `start` (a point of the zonotope; default: the center) in
    /// direction dir leaves the zonotope, a column (n, 1). Not a batch.
    Tensor boundary_point(const Tensor &dir) const;
    Tensor boundary_point(const Tensor &dir, const Tensor &start) const;

    // Measures, predicates and further constructions ---------------------------------------------

    /// The Minkowski sum in which the first min(m, m2, k) generators are added pairwise (the
    /// sums of points with the same coefficients on them), a subset of the plus; k defaults to all.
    Zonotope exact_plus(const Zonotope &Z2, std::optional<int64_t> k = std::nullopt) const;

    /// The Euclidean length of every generator, a row (..., 1, m).
    Tensor generator_length() const;

    /// Always true: a zonotope is bounded.
    bool is_bounded() const;

    /// Whether the generators span the space (their rank is n; singular values up to tol count as
    /// zero), and with the second form also a basis (n, r) of their span (the identity if full).
    /// Not a batch. tol default: the repository tolerance (docs/architecture.md).
    bool is_full_dim(Tol tol = std::nullopt) const;
    std::pair<bool, Tensor> is_full_dim_subspace(Tol tol = std::nullopt) const;

    /// Whether the zonotope equals a zonotope, a box or a point (n, 1): the centers agree (as
    /// within_tol) and so do the generators after compact "all", up to order and sign. Not a batch.
    /// tol default: the repository tolerance.
    bool isequal(const Zonotope &S, Tol tol = std::nullopt) const;
    bool isequal(const Interval &I, Tol tol = std::nullopt) const;
    bool isequal(const Tensor &p, Tol tol = std::nullopt) const;

    /// Whether the zonotope and a zonotope, a box or points (n, N) (all of them) intersect, with S
    /// enlarged by tol (default: the repository tolerance). type "exact" decides by a linear
    /// program, "approx" tests the boxes around the sets. Not a batch.
    bool is_intersecting(const Zonotope &S, const std::string &type = "exact",
                         Tol tol = std::nullopt) const;
    bool is_intersecting(const Interval &I, const std::string &type = "exact",
                         Tol tol = std::nullopt) const;
    bool is_intersecting(const Tensor &p, const std::string &type = "exact",
                         Tol tol = std::nullopt) const;

    /// The zonotope in N dimensions whose state dims[i] is state i (a permutation: new dimensions
    /// would be unbounded). CORA's lift_ projects onto dims, the inverse permutation.
    Zonotope lift(int64_t N, const std::vector<int64_t> &dims) const;

    /// An upper bound (..., 1, 1) of the norm (type 1, 2 or infinity) of the points: mode "ub"
    /// (default) is the norm of the box around the set; "exact" and "ub_convex" are not supported.
    Tensor norm(double type = 2, const std::string &mode = "ub") const;

    /// The distance from the center to the boundary (1, 1) (the largest ball about the center in
    /// the set; 0 if the set is flat) and the nearest point of the boundary (n, 1). Not a batch.
    std::pair<Tensor, Tensor> minnorm() const;

    /// The matrices E, F, A, C of an observer, for the gain "wang-FRad" of intersect_strip.
    struct StripObserver {
        Tensor E, F, A, C;
    };

    /// A zonotope that encloses the intersection with the strips |C x - y| <= phi (C (k, n), phi
    /// and y (k, 1)): center c + L (y - C c), generators [(I - L C) G, L diag(phi)] for a gain L
    /// (n, k) that is given, chosen by the method "normGen" (default), "alamo-FRad" or "bravo" (one
    /// strip), or built from an observer ("wang-FRad"). "svd", "radius" and "alamo-volume" are not
    /// supported.
    Zonotope intersect_strip(const Tensor &C, const Tensor &phi, const Tensor &y,
                             const std::string &method = "normGen") const;
    Zonotope intersect_strip(const Tensor &C, const Tensor &phi, const Tensor &y,
                             const Tensor &lambda) const;
    Zonotope intersect_strip(const Tensor &C, const Tensor &phi, const Tensor &y,
                             const StripObserver &aux) const;

    /// A zonotope that contains the points (n, N): method "maiga" (default) or "stursberg"; with a
    /// frame U (n, n) (orthogonal) the box around the points in that frame.
    static Zonotope enclose_points(const Tensor &points, const std::string &method = "maiga");
    static Zonotope enclose_points(const Tensor &points, const Tensor &U);

    /// A box in the frame whose first axis is the direction (n, 1), as a zonotope, around the
    /// zonotopes (each reduced to order 1), and the inverse of the frame. Not a batch.
    static std::pair<Zonotope, Tensor> enclose_many(const std::vector<Zonotope> &Zs,
                                                    const Tensor &direction);

    /// The Minkowski difference with a zonotope, a box or a point, by scaling the generators:
    /// method "approx" (default), "inner", "exact" (two dimensions, or aligned generators),
    /// "outer", "outer:coarse"; an empty result is an error (no empty set yet). Not a batch.
    Zonotope mink_diff(const Zonotope &subtrahend, const std::string &method = "approx") const;
    Zonotope mink_diff(const Interval &subtrahend, const std::string &method = "approx") const;
    Zonotope mink_diff(const Tensor &p, const std::string &method = "approx") const;

  protected:
    int64_t dim_impl() const override { return c_.shape()[c_.shape().size() - 2]; }
    Tensor center_impl() const override;
    Tensor support_function_impl(const Tensor &d) const override;
    Interval interval_impl() const override;
    Tensor rand_point_impl(int64_t N, Rng &rng) const override;
    Tensor rand_point_impl(int64_t N, Rng &rng, const std::string &type) const;
    Set project_impl(const std::vector<int64_t> &dims) const override;
    std::vector<Polygon> vertices_impl() const override;
    std::vector<std::vector<Point3>> vertices_3d_impl() const override;
    std::string display_impl() const override;

  private:
    Tensor c_, G_;
};

// Operators, as CORA writes them: a matrix M * Z is the linear map, a number s * Z scales, Z + Z2
// is the Minkowski sum, and a column v translates.
inline Zonotope operator*(const Tensor &M, const Zonotope &Z) { return Z.linear_map(M); }
inline Zonotope operator*(const Interval &M, const Zonotope &Z) { return Z.linear_map(M); }
inline Zonotope operator*(double s, const Zonotope &Z) {
    return {Z.center() * s, Z.generators() * s};
}
inline Zonotope operator*(const Zonotope &Z, double s) { return s * Z; }
inline Zonotope operator+(const Zonotope &Z, const Zonotope &Z2) { return Z.plus(Z2); }
inline Zonotope operator+(const Zonotope &Z, const Interval &I) { return Z.plus(I); }
inline Zonotope operator+(const Interval &I, const Zonotope &Z) { return Z.plus(I); }
inline Zonotope operator+(const Zonotope &Z, const Tensor &v) {
    return {Z.center() + v, Z.generators()};
}
inline Zonotope operator+(const Tensor &v, const Zonotope &Z) { return Z + v; }
inline Zonotope operator-(const Zonotope &Z, const Tensor &v) {
    return {Z.center() - v, Z.generators()};
}
inline Zonotope operator-(const Zonotope &Z) { return {Z.center() * -1.0, Z.generators()}; }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
