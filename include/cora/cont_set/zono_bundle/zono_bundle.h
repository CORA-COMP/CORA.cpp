// zono_bundle - the class of zonotope bundles, the intersection of zonotopes, as CORA's zonoBundle
//
// A bundle holds a list of zonotopes of one dimension and one batch shape and denotes the points
// that lie in all of them. Every operation works on the members with the operations of Zonotope
// (never its own copy of them). Immutable: every operation returns a new set. A bundle has at
// least one member (CORA also accepts an empty list, which it takes for the empty set).
//
// Syntax:     ZonoBundle zB({Z1, Z2});   ZonoBundle E = ZonoBundle::empty(n);   ConZonotope cz = zB.con_zonotope();
// Operations: constructor (zono_bundle.cpp), empty, con_zonotope, center, interval, is_bounded,
//             is_full_dim, intersect, cart_prod, enclose, contains, is_intersecting, isequal,
//             lift, enlarge (ContSet), display (one file each). The other operations of the
//             ContSet interface (support_function, rand_point, project, vertices) are
//             Error::not_supported until they are ported.
// Not ported: conPolyZono (needs ConPolyZono), convHull_ and encloseTight (need Polytope).
// See also:   cora/cont_set/cont_set.h, cora/cont_set/zonotope/zonotope.h
//
// CORA: contSet/@zonoBundle/zonoBundle.m

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

class ConZonotope;

class ZonoBundle : public ContSet {
  public:
    static constexpr ClassId kClassId = ClassId::zono_bundle;

    /// The intersection of the zonotopes (CORA's zonoBundle(list)): at least one, one dimension
    /// and one batch shape.
    explicit ZonoBundle(std::vector<Zonotope> zonotopes);

    /// The members (CORA's Z) and their number (CORA's parallelSets).
    const std::vector<Zonotope> &zonotopes() const noexcept { return Z_; }
    int64_t parallel_sets() const noexcept { return int64_t(Z_.size()); }

    /// The empty set of n >= 1 dimensions: the two zonotopes {0} and {e_1}, which share no
    /// point. CORA's has one zonotope whose center has no column, which no operation of a
    /// zonotope can represent.
    static ZonoBundle empty(int64_t n);

    // Conversions ---------------------------------------------------------------------------------

    /// The same set as a constrained zonotope: the first member, intersected exactly with the
    /// others in order (CORA's conZonotope(zB)).
    ConZonotope con_zonotope() const;

    // Queries -------------------------------------------------------------------------------------

    /// Always true: a bundle is an intersection of bounded sets.
    bool is_bounded() const;

    /// Whether the set has a nonempty interior: every member is full-dimensional and so is the
    /// intersection (ConZonotope::is_full_dim, where CORA uses the polytope of the bundle). Not a
    /// batch. tol default: the repository tolerance (docs/architecture.md).
    bool is_full_dim(Tol tol = std::nullopt) const;

    /// Whether the bundle contains the points p (n, N) (all of them), a zonotope, a box or a
    /// bundle, up to tol. For points, method "exact" (default) tests the constrained zonotope of
    /// the bundle, "exact:zonotope" tests the members one by one (both are exact; "approx" is
    /// "exact", as in CORA). A set is contained if every member contains it (the members' own
    /// "exact"); a bundle S is, if each of its facets-wise tests against a member holds. "exact:polytope"
    /// needs Polytope. Not a batch. tol default: the repository tolerance.
    bool contains(const Tensor &p, const std::string &method = "exact", Tol tol = std::nullopt) const;
    bool contains(const Zonotope &S, const std::string &method = "exact",
                  Tol tol = std::nullopt) const;
    bool contains(const Interval &I, const std::string &method = "exact",
                  Tol tol = std::nullopt) const;
    bool contains(const ZonoBundle &S, const std::string &method = "exact",
                  Tol tol = std::nullopt) const;

    /// Whether the bundle and a zonotope, a box, a bundle or points (n, N) (all of them in the
    /// bundle) intersect. type "exact" decides by emptiness of the intersection (a linear
    /// program; tol is used for points only, as in CORA), "approx" tests the boxes around the sets,
    /// S enlarged by tol. Not a batch.
    bool is_intersecting(const ZonoBundle &S, const std::string &type = "exact",
                         Tol tol = std::nullopt) const;
    bool is_intersecting(const Zonotope &Z, const std::string &type = "exact",
                         Tol tol = std::nullopt) const;
    bool is_intersecting(const Interval &I, const std::string &type = "exact",
                         Tol tol = std::nullopt) const;
    bool is_intersecting(const Tensor &p, const std::string &type = "exact",
                         Tol tol = std::nullopt) const;

    /// Whether the bundles are the same set: the members agree one by one (Zonotope::isequal),
    /// else each contains the other. Not a batch. tol default: the repository tolerance.
    bool isequal(const ZonoBundle &S, Tol tol = std::nullopt) const;

    // Combining with other sets -------------------------------------------------------------------

    /// The intersection with a bundle, a zonotope or a box (CORA's and_): exact, the members of
    /// both.
    ZonoBundle intersect(const ZonoBundle &S) const;
    ZonoBundle intersect(const Zonotope &Z) const;
    ZonoBundle intersect(const Interval &I) const;

    /// The Cartesian product with a bundle (every pair of members, this one's the outer index), a
    /// zonotope, a box or a point (n2, 1) (member by member): exact.
    ZonoBundle cart_prod(const ZonoBundle &S) const;
    ZonoBundle cart_prod(const Zonotope &Z) const;
    ZonoBundle cart_prod(const Interval &I) const;
    ZonoBundle cart_prod(const Tensor &p) const;

    /// The Cartesian product of a point p (n1, 1) with a bundle.
    static ZonoBundle cart_prod(const Tensor &p, const ZonoBundle &zB);

    /// A bundle that encloses this one and S (CORA's enclose): member by member, so S needs as
    /// many members (a zonotope counts as many copies). With (M, S) the second set is M * this + S
    /// for a square M (..., n, n): the member i of M * this + S is M Z_i + S for a zonotope S, and
    /// M Z_i + the zonotope of the box around S for a bundle S.
    ZonoBundle enclose(const ZonoBundle &S) const;
    ZonoBundle enclose(const Zonotope &Z) const;
    ZonoBundle enclose(const Tensor &M, const ZonoBundle &S) const;
    ZonoBundle enclose(const Tensor &M, const Zonotope &Zplus) const;

    /// The bundle in N dimensions whose state dims[i] is state i of this one (a permutation of
    /// all N states): member by member Zonotope::lift (CORA's lift_).
    ZonoBundle lift(int64_t N, const std::vector<int64_t> &dims) const;

  protected:
    int64_t dim_impl() const override { return Z_.front().dim(); }
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
    Set enlarge_impl(const Tensor &factor) const override;
    Set lift_impl(int64_t N, const std::vector<int64_t> &proj) const override;
    Set enclose_impl(const ContSet &S2) const override;

  private:
    std::vector<Zonotope> Z_;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
