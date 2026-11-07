// zono_bundle - the class of zonotope bundles, the intersection of zonotopes, as CORA's zonoBundle
//
// A bundle holds a list of zonotopes of one dimension and one batch shape and denotes the points
// that lie in all of them. Every operation works on the members with the operations of Zonotope,
// never with a copy of them. Immutable: every operation returns a new set. A bundle has at least
// one member.
//
// Syntax:     ZonoBundle bundle({Z1, Z2});   ZonoBundle E = ZonoBundle::empty(n);   ConZonotope cz
// = bundle.con_zonotope(); Operations: constructor (zono_bundle.cpp), empty, con_zonotope, center,
// interval, is_bounded,
//             is_full_dim, intersect, cart_prod, enclose, contains, is_intersecting, isequal, lift,
//             enlarge (ContSet), display, origin, from, zonotope, polytope, linear_map, plus,
//             quad_map, reduce, replace, representsa, split, unite, support_function, vertices,
//             rand_point, project, generate_random (one file each)
// Not ported: conPolyZono (needs ConPolyZono), convHull_, volume_, encloseTight, shrink and
//             reduceCombined (need Polytope vertices and volume, and the "methC" reduction)
// See also:   cora/cont_set/cont_set.h, cora/cont_set/zonotope/zonotope.h
//
// CORA: contSet/@zonoBundle/zonoBundle.m
// CORA: contSet/@zonoBundle/dim.m
// CORA authors: Matthias Althoff
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

class ConZonotope;
class Polytope;

class ZonoBundle : public ContSet {
  public:
    static constexpr ClassId kClassId = ClassId::zono_bundle;

    /// The intersection of the zonotopes (CORA's zonoBundle(list)): at least one, of one dimension
    /// and one batch shape.
    explicit ZonoBundle(std::vector<Zonotope> zonotopes);

    /// The members (CORA's Z) and their number (CORA's parallelSets).
    const std::vector<Zonotope> &zonotopes() const noexcept { return Z_; }
    int64_t parallel_sets() const noexcept { return int64_t(Z_.size()); }

    /// The empty set of n >= 1 dimensions: the zonotopes {0} and {e_1}, which share no point.
    /// CORA's has one zonotope whose center has no column.
    static ZonoBundle empty(int64_t n);

    /// The origin {0} in n >= 1 dimensions: one member without generators (CORA's origin).
    static ZonoBundle origin(int64_t n);

    // Conversions ---------------------------------------------------------------------------------

    /// A zonotope or a box as a bundle of one member (CORA's zonotope.zonoBundle,
    /// interval.zonoBundle); static, because a one-element list is also a list of members.
    static ZonoBundle from(const Zonotope &Z);
    static ZonoBundle from(const Interval &I);

    /// The member of least volume, the product of the sums of |G| per row (CORA's zonotope): an
    /// enclosure of the bundle. Not a batch.
    Zonotope zonotope() const;

    /// The halfspace form: the intersection of the polytopes of the members (CORA's polytope).
    Polytope polytope() const;

    /// The same set as a constrained zonotope: the first member intersected exactly with the others
    /// in order (CORA's conZonotope(bundle)).
    ConZonotope con_zonotope() const;

    // Queries -------------------------------------------------------------------------------------

    /// Always true: a bundle is an intersection of bounded sets.
    bool is_bounded() const;

    /// Whether the set has a nonempty interior: every member is full-dimensional and so is the
    /// intersection (decided on its constrained zonotope). Not a batch; tol: the repository one.
    bool is_full_dim(Tol tol = std::nullopt) const;

    /// Whether the bundle contains points (n, N) (all), a zonotope, a box or a bundle, up to tol.
    /// method "exact", "exact:zonotope" or "approx" (as "exact" for points only); not a batch.
    /// "exact:polytope" needs Polytope.
    bool contains(const Tensor &p, const std::string &method = "exact",
                  Tol tol = std::nullopt) const;
    bool contains(const Zonotope &S, const std::string &method = "exact",
                  Tol tol = std::nullopt) const;
    bool contains(const Interval &I, const std::string &method = "exact",
                  Tol tol = std::nullopt) const;
    bool contains(const ZonoBundle &S, const std::string &method = "exact",
                  Tol tol = std::nullopt) const;

    /// Whether the bundle meets a bundle, a zonotope, a box or points (n, N) (all inside). Type
    /// "exact" asks whether the bundle of both is empty; "approx" tests the boxes, S's enlarged by
    /// tol. Not a batch; tol is used for points and "approx" only (as CORA).
    bool is_intersecting(const ZonoBundle &S, const std::string &type = "exact",
                         Tol tol = std::nullopt) const;
    bool is_intersecting(const Zonotope &Z, const std::string &type = "exact",
                         Tol tol = std::nullopt) const;
    bool is_intersecting(const Interval &I, const std::string &type = "exact",
                         Tol tol = std::nullopt) const;
    bool is_intersecting(const Tensor &p, const std::string &type = "exact",
                         Tol tol = std::nullopt) const;

    /// Whether the bundles are the same set: the members agree one by one, else each contains the
    /// other. Not a batch; tol: the repository one.
    bool isequal(const ZonoBundle &S, Tol tol = std::nullopt) const;

    // Combining with other sets -------------------------------------------------------------------

    /// The intersection with a bundle, a zonotope or a box (CORA's and_): exact, the members of
    /// both.
    ZonoBundle intersect(const ZonoBundle &S) const;
    ZonoBundle intersect(const Zonotope &Z) const;
    ZonoBundle intersect(const Interval &I) const;

    /// The Cartesian product with a bundle (every pair of members), a zonotope, a box or a point
    /// (n2, 1), member by member: exact.
    ZonoBundle cart_prod(const ZonoBundle &S) const;
    ZonoBundle cart_prod(const Zonotope &Z) const;
    ZonoBundle cart_prod(const Interval &I) const;
    ZonoBundle cart_prod(const Tensor &p) const;

    /// The Cartesian product of a point p (n1, 1) with a bundle.
    static ZonoBundle cart_prod(const Tensor &p, const ZonoBundle &bundle);

    /// A bundle that encloses this one and S, member by member (S needs as many members).
    ZonoBundle enclose(const ZonoBundle &S) const;
    ZonoBundle enclose(const Zonotope &Z) const;

    /// The same with the second set M this + S for a square M: M Z_i + S for a zonotope S, and
    /// M Z_i + the zonotope of the box around S for a bundle S.
    ZonoBundle enclose(const Tensor &M, const ZonoBundle &S) const;
    ZonoBundle enclose(const Tensor &M, const Zonotope &Zplus) const;

    /// The bundle in N dimensions whose state dims[i] is state i (a permutation of all N states):
    /// every member lifted (CORA's lift_).
    ZonoBundle lift(int64_t N, const std::vector<int64_t> &dims) const;

    // Maps and sums -------------------------------------------------------------------------------

    /// M * bundle for a matrix M (..., k, n) (CORA's mtimes): every member mapped, an enclosure.
    ZonoBundle linear_map(const Tensor &M) const;

    /// The Minkowski sum with a column v (n, 1), a zonotope, a box or a bundle: every member plus
    /// every member of S (k1 k2 members, Althoff and Krogh, CDC 2011, Prop. 2: an enclosure; CORA
    /// adds the box of a bundle S, docs/cora-differences.md).
    ZonoBundle plus(const Tensor &v) const;
    ZonoBundle plus(const Zonotope &Z) const;
    ZonoBundle plus(const Interval &I) const;
    ZonoBundle plus(const ZonoBundle &S) const;

    /// An enclosure of {(x' Q_i x)_i | x in bundle}: every member's quad_map (CORA's quadMap).
    ZonoBundle quad_map(const std::vector<Tensor> &Q) const;

    /// Every member reduced to at most order * n generators (CORA's reduce), method "girard".
    ZonoBundle reduce(int order = 1, const std::string &method = "girard") const;

    /// The bundle with member `index` (0-based) replaced by Z (CORA's replace).
    ZonoBundle replace(int64_t index, const Zonotope &Z) const;

    /// Whether the bundle is also a set of the CORA type ("origin", "interval", "emptySet",
    /// "conZonotope", "polytope", ...); the converted set is not returned. Not a batch.
    bool representsa(const std::string &type, Tol tol = std::nullopt) const;

    /// The two sets (CORA's split(zB, N)) cut at the middle of coordinate N (0-based) of the box
    /// around the bundle: the intersections with the two halves of the box; and for every
    /// coordinate (split(zB)). Not a batch.
    std::pair<ZonoBundle, ZonoBundle> split(int64_t N) const;
    std::vector<std::pair<ZonoBundle, ZonoBundle>> split() const;

    /// An enclosure of the union with a bundle, a zonotope or a box (CORA's or): the union of
    /// the constrained zonotopes (ConZonotope::unite), as a bundle. Not a batch.
    ZonoBundle unite(const ZonoBundle &S) const;
    ZonoBundle unite(const Zonotope &Z) const;
    ZonoBundle unite(const Interval &I) const;

    using ContSet::rand_point;

    /// N random points as columns (n, N): type "standard" or "extreme", as ConZonotope::rand_point
    /// of the constrained zonotope. Not a batch.
    Tensor rand_point(int64_t N, Rng &rng, const std::string &type) const;

    /// A random bundle, as CORA's generateRandom: dimension (default 1..10) and nr_zonotopes
    /// (default 2..5), all members random zonotopes about one center, each moved so that the
    /// center stays inside.
    static ZonoBundle generate_random(Rng &rng, std::optional<int64_t> dimension = std::nullopt,
                                      std::optional<int64_t> nr_zonotopes = std::nullopt);

  protected:
    // The unchecked implementations of the ContSet interface and its generic operations.
    int64_t dim_impl() const override { return Z_.front().dim(); }
    Tensor center_impl() const override;
    Tensor support_function_impl(const Tensor &d) const override;
    Set linear_map_impl(const Tensor &M) const override;
    Set translate_impl(const Tensor &v) const override;
    Set reduce_impl(int order) const override;
    Interval interval_impl() const override;
    Tensor rand_point_impl(int64_t N, Rng &rng) const override;
    Set project_impl(const std::vector<int64_t> &dims) const override;
    std::vector<Polygon> vertices_impl() const override;
    std::vector<std::vector<Point3>> vertices_3d_impl() const override;
    std::string display_impl() const override;

    bool is_bounded_impl() const override;
    bool is_full_dim_impl(Tol tol) const override;
    Set enlarge_impl(const Tensor &factor) const override;
    Set lift_impl(int64_t N, const std::vector<int64_t> &proj) const override;
    Set enclose_impl(const ContSet &S2) const override;

  private:
    std::vector<Zonotope> Z_;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
