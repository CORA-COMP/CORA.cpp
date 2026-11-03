// fullspace - the class of the whole space R^n, as CORA's fullspace
//
// A set that holds only its dimension n. Immutable: every operation returns a new set. n = 0 is
// allowed, as in CORA, which refuses the operations that need a point of R^0
// (Error::not_supported). A point p is the interval [p, p] wherever a set is an operand.
//
// Syntax:     Fullspace F(n);   Fullspace F = Fullspace::inf(n);   Set R = M * F;
// Operations: inf, box, intersect, plus, convex_hull, contains, is_intersecting, isequal,
//             is_bounded, is_full_dim, enclose_points, lift, linear_map, center, interval (one
//             file each); the constructor checks its input (fullspace.cpp), which also holds what
//             ContSet needs of every class (support_function, rand_point, project, vertices,
//             display)
// See also:   cora/cont_set/cont_set.h, cora/cont_set/interval/interval.h
//
// CORA: contSet/@fullspace/fullspace.m

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

class Fullspace : public detail::DimensionSet {
  public:
    static constexpr ClassId kClassId = ClassId::fullspace;

    /// The whole space of n >= 0 dimensions.
    explicit Fullspace(int64_t n);

    /// The whole space of n dimensions (CORA's fullspace.Inf).
    static Fullspace inf(int64_t n);

    /// The smallest box around the set (CORA's box): the set itself.
    Fullspace box() const;

    /// The intersection with S (CORA's and_): S.
    Set intersect(const Set &S) const;

    /// The Minkowski sum with S: the whole space. An empty interval S is Error::not_supported,
    /// as the sum would be an empty set and there is no such class yet.
    Fullspace plus(const ContSet &S) const;

    /// The convex hull with S: the whole space.
    Fullspace convex_hull(const ContSet &S) const;

    /// Whether the set contains S: always. tol relaxes nothing, but is checked; default: the
    /// repository tolerance (docs/architecture.md). Error::not_supported in R^0.
    bool contains(const ContSet &S, Tol tol = std::nullopt) const;

    /// Whether the set meets S: unless S is an empty interval. tol as contains.
    /// Error::not_supported in R^0.
    bool is_intersecting(const ContSet &S, Tol tol = std::nullopt) const;

    /// Whether S is the whole space of the same dimension: a Fullspace, or an Interval with all
    /// bounds infinite; false for any other set, for a point and for a different dimension. tol
    /// as contains.
    bool isequal(const ContSet &S, Tol tol = std::nullopt) const;

    /// Whether the set is bounded: only in R^0.
    bool is_bounded() const;

    /// Whether the set has a nonempty interior: always.
    bool is_full_dim() const;

    /// The whole space of the dimension of the points (..., n, N) (CORA's enclosePoints).
    static Fullspace enclose_points(const Tensor &points);

    /// The set in N >= n dimensions, the new ones unbounded (CORA's lift_): the whole space of N
    /// dimensions. dims (0-based, one position per dimension) is checked and otherwise unused.
    Fullspace lift(int64_t N, const std::vector<int64_t> &dims) const;

    /// The image under a matrix M (CORA's mtimes): the whole space of dimension m for M with m
    /// rows and no zero row, an interval with the zero rows at 0 and the others unbounded if M
    /// has zero rows, a point at 0 for a zero number. It is an enclosure when M has not full row
    /// rank (CORA's own). A 1 x 1 matrix is a scalar. A matrix of intervals is
    /// Error::noops, as in CORA.
    Set linear_map(const Tensor &M) const;
    Set linear_map(const Interval &) const;

  protected:
    Tensor center_impl() const override;
    Tensor support_function_impl(const Tensor &d) const override;
    Interval interval_impl() const override;
    Tensor rand_point_impl(int64_t N, Rng &rng) const override;
    Set project_impl(const std::vector<int64_t> &dims) const override;
    std::vector<Polygon> vertices_impl() const override;
    std::vector<std::vector<Point3>> vertices_3d_impl() const override;
    std::string display_impl() const override;

};

/// The image of the whole space under a matrix, as CORA writes M * F.
inline Set operator*(const Tensor &M, const Fullspace &F) { return F.linear_map(M); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
