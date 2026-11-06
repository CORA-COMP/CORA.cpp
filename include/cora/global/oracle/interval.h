// interval - lean::Interval, a box (or interval matrix) of exact values, and its oracle operations
//
// Syntax:     lean::Interval I(inf, sup);   lean::Interval J = I.plus(I2).sin();   J.gather();
// Values:     the bounds are exact; an operation returns the box the oracle proves to enclose the
//             real result (outward rounded), so a box has no separate error part
// Operations: one method per `interval.*` op of the oracle, named as CORA.cpp names CORA's: `and`
//             is intersect, `or` is unite, `mtimes` is linear_map, `Inf` is unbounded
// Options:    CORA's option names, 0-based indices (as the oracle); `p` and `k` tune the elementary
//             functions (CORA has no such argument); `tol` is accepted at 0 only
// See also:   lean/tensor.h, lean/zonotope.h, lean/request.h
//
// Authors:       Tobias Ladner, Claude Sonnet 5.5, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#pragma once

#include "cora/cont_set/interval/interval.h"
#include "cora/global/oracle/tensor.h"

#include <optional>
#include <utility>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

class Zonotope;
struct Rounded;
struct Vertices;

/// An answer of the oracle: `value` as CORA's operation would answer, a proof when `proved`
/// (otherwise the oracle found none, which says nothing).
struct Certified {
    bool value, proved;
};

/// A box of lean values: inf <= x <= sup.
class Interval {
  public:
    Interval(Tensor inf, Tensor sup);

    /// The exact box of a cora::Interval (values in the current dtype).
    Interval(const cora::Interval &I);

    const Tensor &inf() const noexcept { return inf_; }
    const Tensor &sup() const noexcept { return sup_; }

    /// The exact interval of double bounds (values in the current dtype).
    static Interval from(const cora::Tensor &inf, const cora::Tensor &sup);

    /// Back to double bounds; throws if a bound is not a double.
    std::pair<cora::Tensor, cora::Tensor> gather() const;

    Json to_json() const;
    static Interval from_json(const std::string &dtype, const Json &j);

    // Construction -----------------------------------------------------------------------------

    /// The empty set, the origin and the unbounded box (CORA's Inf) in n dimensions.
    static Interval empty(int64_t n);
    static Interval origin(int64_t n);
    static Interval unbounded(int64_t n);

    /// The box enclosing the columns of `points`; the interval matrix [c - r, c + r].
    static Interval enclose_points(const Tensor &points);
    static Interval interval_matrix(const Tensor &c, const Tensor &r);

    // Elementwise ------------------------------------------------------------------------------

    Interval abs() const;
    Interval uminus() const;
    Interval uplus() const;
    Interval sign() const;
    Interval round() const;
    Interval sqrt(int p = 20) const;
    Interval exp(int p = 20) const;
    Interval sin(int p = 20) const;
    Interval cos(int p = 20) const;
    Interval tan(int p = 20) const;
    Interval sinh(int p = 20) const;
    Interval cosh(int p = 20) const;
    Interval tanh(int p = 20) const;
    Interval log(int p = 20, int k = 20) const;
    Interval asin(int p = 20, int k = 20) const;
    Interval acos(int p = 20, int k = 20) const;
    Interval atan(int p = 20, int k = 20) const;
    Interval asinh(int p = 20, int k = 20) const;
    Interval acosh(int p = 20, int k = 20) const;
    Interval atanh(int p = 20, int k = 20) const;
    // atan2 of two intervals
    Interval atan2(const Interval &I2, int p = 20, int k = 20) const;

    // Scaling, powers and quotients.
    Interval enlarge(double factor) const;
    Interval power(int64_t exponent) const;
    Interval rdivide(const Interval &I2) const;
    Interval mrdivide(double d) const;

    // Arithmetic and set operations ------------------------------------------------------------

    Interval plus(const Interval &I2) const;
    Interval minus(const Interval &I2) const;
    Interval times(const Interval &I2) const;
    Interval unite(const Interval &I2) const;
    Interval convex_hull(const Interval &I2) const;
    Interval mink_diff(const Interval &I2) const;
    Interval min(const Interval &I2) const;
    Interval max(const Interval &I2) const;
    Interval intersect(const Interval &I2) const;
    Interval enclose(const Interval &I2) const;
    Interval cart_prod(const Interval &I2) const;

    /// M * I for a matrix M, or for an interval matrix [M]; I is a box or an interval matrix.
    Interval linear_map(const Tensor &M) const;
    Interval linear_map(const Interval &M) const;

    /// The enclosure of x' Q_i x over the box, one row per matrix of Q.
    Interval quad_map(const std::vector<Tensor> &Q) const;

    // Predicates (exact answers) ---------------------------------------------------------------

    bool le(const Interval &I2, double tol = 0) const;
    bool lt(const Interval &I2, double tol = 0) const;
    bool isequal(const Interval &I2, double tol = 0) const;
    bool is_full_dim(double tol = 0) const;
    bool is_bounded(double tol = 0) const;

    /// type "emptySet", "point" or "origin".
    bool representsa(const std::string &type, double tol = 0) const;

    // Certified predicates: contains proves the containment, is_intersecting the disjointness ---

    Certified contains(const Interval &I2, double tol = 0) const;
    Certified contains(const Zonotope &Z, double tol = 0) const;
    Certified is_intersecting(const Interval &I2, double tol = 0) const;
    Certified is_intersecting(const Zonotope &Z, double tol = 0) const;

    // Numbers ----------------------------------------------------------------------------------

    Tensor center() const;
    Tensor rad() const;
    Tensor infimum() const;
    Tensor supremum() const;

    /// An upper bound (type 2, mode "ub") of the radius, the norm and the volume; the support
    /// function along `dir` (type "upper").
    double radius(int type = 2, const std::string &mode = "ub") const;
    double norm(int type = 2, const std::string &mode = "ub") const;
    double volume() const;
    double support_function(const Tensor &dir, const std::string &type = "upper") const;

    // Selection and reshaping ------------------------------------------------------------------

    Interval project(const std::vector<int64_t> &dims) const;
    Interval lift(int64_t N, const std::vector<int64_t> &dims) const;
    Interval project_high_dim(int64_t N, const std::vector<int64_t> &dims) const;
    std::pair<Interval, Interval> split(int64_t dim) const;
    Interval partition(int64_t splits, const std::vector<int64_t> &ind) const;
    Interval dirac(int64_t n) const;

    /// Where the ray from `start` along `dir` leaves the box, nothing if it stays inside.
    std::optional<Interval> boundary_point(const Tensor &dir, const Tensor &start) const;

    /// The box of the projection onto the hyperplane {x | c'x = d} for a unit normal c.
    Interval project_on_hyperplane(const Tensor &c, double d) const;

    // Matrices ---------------------------------------------------------------------------------

    Interval transpose() const;
    Interval ctranspose() const;
    Interval prod() const;
    Interval diag() const;
    Interval kron(const Interval &I2) const;
    Interval horzcat(const Interval &I2) const;
    Interval vertcat(const Interval &I2) const;
    Interval sum(int dim) const;
    Interval cat(const Interval &I2, int dim) const;
    Interval mpower(int64_t exponent) const;
    Interval tril(int64_t K = 0) const;
    Interval triu(int64_t K = 0) const;

    // Other classes ----------------------------------------------------------------------------

    /// Order reduction to `order` with method "girard" (a box stays a box).
    Interval reduce(int order, const std::string &method = "girard") const;

    /// The box as a zonotope, and its 2^n corners with an error box.
    Zonotope zonotope() const;
    Vertices vertices() const;

  private:
    Tensor inf_, sup_;
};

/// The result of round_to: the rounded values and an interval enclosing x - value.
struct Rounded {
    Tensor value;
    Interval error;
};

/// The vertices of a set as columns of V, and a box E that encloses the rounding errors.
struct Vertices {
    Tensor V;
    Interval E;
};

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
