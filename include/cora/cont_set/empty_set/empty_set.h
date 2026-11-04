// empty_set - the class of the empty set of n dimensions, as CORA's emptySet
//
// A set without points: its only data is the dimension n. Immutable: every operation returns a
// new set. It has no batch (it holds no tensor) and, as in CORA, its center and its random
// points have no columns: shape (n, 0).
//
// Syntax:     EmptySet E(n);   EmptySet E = EmptySet::empty(n);   EmptySet F = M * E;
// Operators:  M * E (linear_map), s * E, as in CORA
// Operations: intersect, plus, convex_hull, contains, is_intersecting, isequal, is_bounded,
//             is_full_dim, linear_map, lift, project_high_dim, complement, radius, volume,
//             representsa, generate_random, empty, center, support_function, interval, rand_point,
//             project, vertices, vertices_3d, display (one file each); the constructor checks its
//             input (empty_set.cpp)
// See also:   cora/cont_set/cont_set.h, cora/cont_set/interval/interval.h
//
// CORA: contSet/@emptySet/emptySet.m

#pragma once

#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/dimension_set.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"
#include "cora/global/tensor/tolerance.h"

#include <optional>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

class Fullspace;
class Rng;

class EmptySet : public detail::DimensionSet {
  public:
    static constexpr ClassId kClassId = ClassId::empty_set;

    /// The empty set of n >= 0 dimensions.
    explicit EmptySet(int64_t n);

    /// The empty set of n dimensions (CORA's emptySet.empty); n = 0 by default.
    static EmptySet empty(int64_t n = 0);

    /// Whether the set is bounded: always.
    bool is_bounded() const;

    /// The empty set of dimension n, drawn as CORA's generateRandom does: given, or a random
    /// integer in 1..10 (CORA's randi(10)) taken from rng.
    static EmptySet generate_random(Rng &rng, std::optional<int64_t> dimension = std::nullopt);

    /// The radius of the smallest ball around the set: 0, the supremum over no point (CORA's
    /// radius).
    double radius() const;

    /// The volume: 0 (CORA's volume_).
    double volume() const;

    /// The complement (CORA's not; `not` is a C++ keyword): the whole space of the dimension.
    Fullspace complement() const;

    /// Whether the set is also a set of CORA's type ("origin", "zonotope", "fullspace", ...)
    /// (CORA's representsa_): false for "origin", "point", "halfspace", "hyperplane",
    /// "parallelotope", "probZonotope" and "fullspace", true for the others. "levelSet" and
    /// "polygon" are Error::not_supported. tol is checked and relaxes nothing; default: the
    /// repository tolerance (docs/architecture.md).
    bool representsa(const std::string &type, Tol tol = std::nullopt) const;

    // Combining with other sets -------------------------------------------------------------------

    /// The intersection with any set of the dimension (CORA's and_): the empty set.
    EmptySet intersect(const ContSet &S) const;

    /// The Minkowski sum with a set, or a translation by a column v (..., n, 1): the empty set.
    EmptySet plus(const ContSet &S) const;
    EmptySet plus(const Tensor &v) const;

    /// The convex hull with S (CORA's convHull_): S itself.
    Set convex_hull(const Set &S) const;

    /// Whether the set contains S: true exactly if S is empty (a set without center columns), up
    /// to tol as CORA passes it on. A point matrix p (n, N) is contained if it has no columns.
    bool contains(const ContSet &S, Tol tol = std::nullopt) const;
    bool contains(const Tensor &p, Tol tol = std::nullopt) const;

    /// Whether S meets the set (CORA's isIntersecting_): never.
    bool is_intersecting(const ContSet &S, Tol tol = std::nullopt) const;

    /// Whether S is the empty set of this dimension (a set without center columns); tol is CORA's
    /// argument (default: the repository tolerance) and relaxes nothing here.
    bool isequal(const ContSet &S, Tol tol = std::nullopt) const;

    // Maps and embeddings -------------------------------------------------------------------------

    /// M * E for a matrix M (..., k, n), a number or an interval matrix: the empty set of k
    /// dimensions (CORA's mtimes).
    EmptySet linear_map(const Tensor &M) const;
    EmptySet linear_map(const Interval &M) const;
    EmptySet linear_map(double s) const;

    /// The empty set in N dimensions (CORA's lift_); dims are the coordinates of this set's
    /// dimensions in the larger space, 0-based.
    EmptySet lift(int64_t N, const std::vector<int64_t> &dims) const;

    /// The empty set in N dimensions (CORA's projectHighDim_), with the checks of lift.
    EmptySet project_high_dim(int64_t N, const std::vector<int64_t> &proj) const;

  protected:
    Tensor center_impl() const override;
    Tensor support_function_impl(const Tensor &d) const override;
    Interval interval_impl() const override;
    Tensor rand_point_impl(int64_t N, Rng &rng) const override;
    Set project_impl(const std::vector<int64_t> &dims) const override;
    std::vector<Polygon> vertices_impl() const override;
    std::vector<std::vector<Point3>> vertices_3d_impl() const override;
    std::string display_impl() const override;
    Set linear_map_impl(const Tensor &M) const override;
    Set translate_impl(const Tensor &v) const override;
    Set compact_impl(const std::optional<std::string> &method, Tol tol) const override;
    bool is_full_dim_impl(Tol tol) const override;
    Set lift_impl(int64_t N, const std::vector<int64_t> &proj) const override;
    Tensor norm_impl(Tensor::Norm type, const std::string &mode) const override;
};

// Operators, as CORA writes them: a matrix or a number times the empty set.
inline EmptySet operator*(const Tensor &M, const EmptySet &E) { return E.linear_map(M); }
inline EmptySet operator*(const Interval &M, const EmptySet &E) { return E.linear_map(M); }
inline EmptySet operator*(double s, const EmptySet &E) { return E.linear_map(s); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
