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
//             enlarge (ContSet), display (one file each); the rest of the ContSet interface
//             (support_function, rand_point, project, vertices) is Error::not_supported until
//             ported
// Not ported: conPolyZono (needs ConPolyZono), convHull_ and encloseTight (need Polytope)
// See also:   cora/cont_set/cont_set.h, cora/cont_set/zonotope/zonotope.h
//
// CORA: contSet/@zonoBundle/zonoBundle.m
//
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
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

class ConZonotope;

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

    // Conversions ---------------------------------------------------------------------------------

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
