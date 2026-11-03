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
//             enclose_many, mink_diff, origin, radius, rank, representsa, project_high_dim,
//             rotate, quad_map, split, split_first_gen, split_longest_gen, reduce_under_approx,
//             unite, tensor_multiplication, tensor_multiplication_zono, underapproximate, volume,
//             zonotope_norm (one file each, as in CORA's @zonotope); the constructor checks its
//             input (zonotope.cpp)
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

    // Embedding maps and cutting ------------------------------------------------------------------

    /// The origin {0} in n dimensions (no generators), n >= 1.
    static Zonotope origin(int64_t n);

    /// The zonotope in N dimensions whose state dims[i] is state i of this one and zero elsewhere
    /// (CORA's projectHighDim_); dims lists positions 0..N-1, a repeated one keeps the last.
    Zonotope project_high_dim(int64_t N, const std::vector<int64_t> &dims) const;

    /// The rotation by angle (radians) in the plane of the two states dims (0-based, different):
    /// the center and every generator turn by [cos -sin; sin cos] there. Exact in real numbers.
    Zonotope rotate(const std::vector<int64_t> &dims, double angle) const;

    /// An upper bound (..., 1, 1) of the distance from the center: the smaller of the sum of the
    /// generator lengths and the 2-norm of the radius of the box around the zonotope.
    Tensor radius() const;

    /// The rank of the generators (singular values up to tol count as zero). Not a batch. tol
    /// default: the repository tolerance (docs/architecture.md; CORA's own default is 0).
    int64_t rank(Tol tol = std::nullopt) const;

    /// Whether the zonotope is also a set of the CORA type ("origin", "point", "interval",
    /// "parallelotope", "capsule", "zonotope", ...), judged after dropping the zero generators;
    /// the converted set (CORA's second output) is not returned. "conHyperplane" and "ellipsoid"
    /// are Error::not_supported, as in CORA; "polygon" has no case in CORA and is refused too.
    /// Not a batch. tol default: the repository tolerance.
    bool representsa(const std::string &type, Tol tol = std::nullopt) const;

    /// An enclosure of {(x' Q_i x)_i | x in Z}: Q[i] are the (n, n) matrices of the outputs. Its
    /// center and generators are those of CORA's quadMap (squares, then the mixed terms); with at
    /// most one nonzero Q a single box generator. (matZonotope entries of Q are not supported.)
    Zonotope quad_map(const std::vector<Tensor> &Q) const;

    /// An enclosure of {(x1' Q_i x2)_i | x1 in Z, x2 in Z2}.
    Zonotope quad_map(const Zonotope &Z2, const std::vector<Tensor> &Q) const;

    /// The two halves of the box around the zonotope cut at the center of coordinate N (0-based)
    /// (CORA's split(Z, N)), and the list for every coordinate (split(Z)). The pieces enclose Z.
    /// CORA indexes the nonzero generators of the box, so a flat coordinate shifts or breaks it.
    std::pair<Zonotope, Zonotope> split(int64_t N) const;
    std::vector<std::pair<Zonotope, Zonotope>> split() const;

    /// The two halves along generator N (0-based) of the zonotope reduced to at most max_order
    /// (CORA's split(Z, N, maxOrder)): centers c -+ g_N/2, that generator halved. They cover Z.
    std::pair<Zonotope, Zonotope> split(int64_t N, int max_order) const;

    /// The pieces cut off by a plane through the zonotope normal to dir (CORA's split(Z, dir)):
    /// the generators lose their share along dir, which becomes the one generator of the halves.
    /// Not a batch.
    std::pair<Zonotope, Zonotope> split(const Tensor &dir) const;

    /// As above for the direction aux_dir with the part along orig_dir removed (CORA's
    /// split(Z, origDir, auxDir)).
    std::pair<Zonotope, Zonotope> split(const Tensor &orig_dir, const Tensor &aux_dir) const;

    /// The pieces on the two sides of the plane a'x = d (CORA's split(Z, halfspace)): both halves
    /// of the direction split at that plane, or the zonotope alone if the plane misses it. Not a
    /// batch.
    std::vector<Zonotope> split(const Tensor &a, double d) const;

    /// The two halves along the generator g_j of largest sum_k |g_j' g_k| (CORA's splitFirstGen,
    /// which passes the index to a split by coordinate; CORA.cpp splits that generator). No
    /// generator: the zonotope twice. Not a batch.
    std::pair<Zonotope, Zonotope> split_first_gen() const;

    /// The two halves along the longest generator, centers c + g/2 first (CORA's splitLongestGen);
    /// no generator: the zonotope twice. Not a batch.
    std::pair<Zonotope, Zonotope> split_longest_gen() const;

    /// A zonotope inside this one with at most order * n generators (CORA's reduceUnderApprox),
    /// method "yang" (merge the closest pair by sum or difference), "raghuraman" or "kochdumper"
    /// (also "sum"). A zonotope with few enough generators is returned without its zero
    /// generators. "sadraddini", "scale", "boxLP", "boxCone", "boxNLP", "boxWetzlinger", "nlp",
    /// "cluster" (and the old names) are Error::not_supported. Not a batch.
    Zonotope reduce_under_approx(const std::string &method, double order) const;

    /// A zonotope that contains the union with the sets Zs (CORA's or): method "linprog" (default;
    /// the smallest generator scaling and center that cover the iterative enclosure's facets),
    /// "iterative" (pairwise enclose) or "parallelotope" (a box in the principal axes of the
    /// generators). "tedrake" and "althoff", and an order reduction, are Error::not_supported.
    /// Not a batch.
    Zonotope unite(const std::vector<Zonotope> &Zs, const std::string &method = "linprog",
                   std::optional<int> order = std::nullopt) const;
    Zonotope unite(const Zonotope &Z2, const std::string &method = "linprog",
                   std::optional<int> order = std::nullopt) const;
    Zonotope unite(const Interval &I, const std::string &method = "linprog",
                   std::optional<int> order = std::nullopt) const;
    Zonotope unite(const Tensor &p, const std::string &method = "linprog",
                   std::optional<int> order = std::nullopt) const;

    // Tensor images and measures -----------------------------------------------------------------

    /// The lists of tensor_multiplication, as CORA's options.list: perm (P, 3) the ordered triples
    /// of generators, comb (C, 3) the unordered ones (only its row count C is read: the number of
    /// generators of the result), map (P) the column in [0, C) that triple p adds to.
    struct TensorLists {
        std::vector<std::vector<int64_t>> perm, comb;
        std::vector<int64_t> map;
    };

    /// A zonotope (0, H) that encloses {sum_jkl M[i, j, k, l] x_j x_k x_l | x in Z} for a zonotope
    /// centered at the origin and a third-order tensor M (n, n, n, n); column map[p] of H sums the
    /// products of the generators perm[p]. The second-order form fails in CORA for every input and
    /// is not supported (docs/cora-differences.md).
    Zonotope tensor_multiplication(const Tensor &M, const TensorLists &list) const;

    /// The same for a tensor in a zonotope {M_c + sum_i b_i M_i | |b|_inf <= 1} (M_c and the M_i
    /// third-order tensors): the Minkowski sum of the results for M_c and for every M_i.
    Zonotope tensor_multiplication_zono(const Tensor &M_center,
                                        const std::vector<Tensor> &M_generators,
                                        const TensorLists &list) const;

    /// The vertices c +- sum_j sign(s_i' g_j) g_j of the zonotope for the directions s_i, the
    /// columns of S (n, k), as the columns (..., n, 2k) in the order +, - per direction; a subset
    /// of the zonotope. Without S: the generators if there are at least as many as dimensions,
    /// else dominant_directions (not a batch).
    Tensor underapproximate() const;
    Tensor underapproximate(const Tensor &S) const;

    /// The volume (..., 1, 1): method "exact" (default) is 2^n times the sum of |det| over the
    /// n-subsets of generators; "alamo" is 2^n sqrt(det(G G')), a lower bound; both are 0 if the
    /// generators do not span the space. "reduce" (order reduction by 'pca') is not supported.
    /// order is the one of "reduce".
    Tensor volume(const std::string &method = "exact", int order = 5) const;

    /// The norm (1, 1) of the point p (n, 1) w.r.t. the zonotope centered at the origin (its
    /// center is ignored): min s with G beta = p, |beta|_inf <= s, by a linear program; infinity if
    /// p is not in the span of the generators. Not a batch.
    Tensor zonotope_norm(const Tensor &p) const;

    /// The norm and the minimizer beta (m, 1) (an empty (0, 1) if the norm is infinite); a
    /// warning if the center is not the origin, as the minimizer is then not valid.
    std::pair<Tensor, Tensor> zonotope_norm_minimizer(const Tensor &p) const;

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

    // The generic operations of ContSet, which a zonotope answers by its own (defined beside them).
    Set translate_impl(const Tensor &v) const override;
    Set compact_impl(const std::optional<std::string> &method, Tol tol) const override;
    std::optional<std::vector<std::string>> compact_methods_impl() const override;
    Set minkowski_difference_impl(const ContSet &S2,
                                  const std::optional<std::string> &type) const override;
    bool is_bounded_impl() const override;
    bool is_full_dim_impl(Tol tol) const override;
    Set reduce_impl(int order) const override;
    Set linear_map_impl(const Tensor &M) const override;

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
/// -Z as CORA's uminus: centre and generators negated, like ContSet::uminus.
inline Zonotope operator-(const Zonotope &Z) { return {Z.center() * -1.0, Z.generators() * -1.0}; }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
