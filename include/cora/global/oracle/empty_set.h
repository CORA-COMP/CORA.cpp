// empty_set - lean::EmptySet, the empty set of n dimensions, and its oracle operations
//
// Syntax:     lean::EmptySet E(3);   bool b = E.contains(S);   lean::EmptySet F = E.lift(5, dims);
// Values:     the set is its dimension; it holds no scalar, so every answer is exact
// Operations: one method per `emptySet.*` op of the oracle, named as CORA.cpp names CORA's. The
//             other operand of an op is a cora set: the oracle reads its dimension (plus, isequal)
//             and whether it is empty, a center without columns (contains, isIntersecting)
// See also:   lean/request.h, lean/interval.h

#pragma once

#include "cora/cont_set/cont_set.h"
#include "cora/global/oracle/interval.h"
#include "cora/global/oracle/zonotope.h"

#include <optional>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

class Fullspace;

class EmptySet {
  public:
    explicit EmptySet(int64_t n) : n_(n) {}

    int64_t dim() const noexcept { return n_; }

    /// The empty set of n dimensions.
    static EmptySet empty(int64_t n);

    /// The set operations of CORA's and_ and plus: the empty set, for a set S of the dimension.
    EmptySet intersect(const cora::ContSet &S) const;
    EmptySet plus(const cora::ContSet &S) const;

    /// The convex hull with a box or a zonotope: that set itself.
    Interval convex_hull(const Interval &I) const;
    Zonotope convex_hull(const Zonotope &Z) const;

    /// Whether S is contained (true iff S is empty), whether S meets the set (never), and whether
    /// S is the empty set of the same dimension.
    bool contains(const cora::ContSet &S, double tol = 0) const;
    bool is_intersecting(const cora::ContSet &S, double tol = 0) const;
    bool isequal(const cora::ContSet &S, double tol = 0) const;

    /// The image under a matrix (m, n): the empty set of m dimensions; the support function:
    /// -Inf.
    EmptySet linear_map(const Tensor &M) const;
    double support_function(const Tensor &dir, const std::string &type = "upper") const;

    bool is_bounded(double tol = 0) const;
    bool is_full_dim(double tol = 0) const;

    /// The center: nothing, as the set has no point.
    std::optional<Tensor> center() const;

    /// The interval hull (the crossed box of CORALean's Interval.empty), and the vertices (n, 0).
    Interval interval() const;
    Tensor vertices() const;

    /// The set in N dimensions at the coordinates dims (0-based), and the projection onto dims.
    EmptySet lift(int64_t N, const std::vector<int64_t> &dims) const;
    EmptySet project(const std::vector<int64_t> &dims) const;
    EmptySet project_high_dim(int64_t N, const std::vector<int64_t> &proj) const;

    /// The radius and the volume (0, 0), and the complement (the whole space of the dimension).
    double radius() const;
    double volume() const;
    Fullspace complement() const;

  private:
    int64_t n_;
};

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
