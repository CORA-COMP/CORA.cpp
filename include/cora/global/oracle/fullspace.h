// fullspace - lean::Fullspace, the whole space R^n, and the oracle's `fullspace.*` operations
//
// Syntax:     lean::Fullspace F(n);   F.intersect(I);   F.is_bounded();   F.lift(N, dims);
// Values:     the set holds only n; every result is exact, as CORALean's definitions use no scalar
// Operations: one method per `fullspace.*` op of the oracle, named as CORA.cpp names CORA's (`and`
//             is intersect, `Inf` is inf, `convHull` is convex_hull); `tol` is accepted at 0 only
// See also:   lean/interval.h, lean/request.h, cora/cont_set/fullspace/fullspace.h
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#pragma once

#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/global/oracle/interval.h"
#include "cora/global/oracle/tensor.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

class EmptySet;

class Fullspace {
  public:
    explicit Fullspace(int64_t n) : n_(n) {}
    Fullspace(const cora::Fullspace &F) : n_(F.dim()) {}

    /// The dimension.
    int64_t dim() const noexcept { return n_; }

    Json to_json() const;
    static Fullspace from_json(const std::string &dtype, const Json &j);

    /// R^n, and the oracle's box of it.
    static Fullspace inf(int64_t n);
    Fullspace box() const;

    /// The set and its interval hull (-Inf, Inf)^n; the empty interval at n = 0.
    Interval interval() const;

    /// The intersection with a set: that set. The Minkowski sum, the convex hull: R^n.
    Fullspace intersect(const Fullspace &F2) const;
    Interval intersect(const Interval &I) const;
    Fullspace plus(const Fullspace &F2) const;
    Fullspace convex_hull(const Fullspace &F2) const;

    /// Whether the set contains / meets a box: contains proves the containment, is_intersecting
    /// the disjointness.
    Certified contains(const Interval &I) const;
    Certified is_intersecting(const Interval &I) const;
    bool isequal(const Fullspace &F2, double tol = 0) const;

    /// The image under a matrix (m, n): the whole space of m dimensions; the support function:
    /// Inf, 0 for the zero direction.
    Fullspace linear_map(const Tensor &M) const;
    double support_function(const Tensor &dir, const std::string &type = "upper") const;

    /// Whether the set is also a set of the type "emptySet", "point", "origin" or "fullspace":
    /// true for "fullspace" only (n > 0). The corners of [-Inf, Inf]^n as the columns of V
    /// (n, 2^n), for n up to 20.
    bool representsa(const std::string &type, double tol = 0) const;
    Tensor vertices() const;

    bool is_bounded() const;
    bool is_full_dim() const;
    Tensor center() const;

    /// R^n of the dimension of the points (n, N), and R^N.
    static Fullspace enclose_points(const Tensor &points);
    Fullspace lift(int64_t N, const std::vector<int64_t> &dims) const;

    /// The projection onto dims, and the embedding into N dimensions (proj as dims of lift).
    Fullspace project(const std::vector<int64_t> &dims) const;
    Fullspace project_high_dim(int64_t N, const std::vector<int64_t> &proj) const;

    /// The radius and the volume (Inf, Inf for 0 < n), and the complement (the empty set).
    double radius() const;
    double volume() const;
    EmptySet complement() const;

  private:
    int64_t n_;
};

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
