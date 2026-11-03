// interval - the class of intervals I = {x | inf <= x <= sup}, as CORA's interval
//
// A box with column bounds (..., n, 1). As in CORA the bounds may be matrices (..., n, n): an
// interval matrix is an Interval too, and Zonotope::linear_map takes it. Immutable: every
// operation returns a new interval.
//
// Syntax:     Interval I(inf, sup);   Interval I = Interval::generate_random(n, rng);
// Operators:  I + I2, I + v, I - I2, I - v, v - I, -I, s * I, M * I, I / s, I <= I2, I < I2, as
//             in CORA
// Operations: support_function, linear_map, plus, rand_point, contains, stack, project, vertices,
//             generate_random, display, center, abs, intersect, cart_prod, cat, unbounded,
//             boundary_point, acos, asin, atan, atan2, acosh, asinh, atanh, cos, cosh, exp,
//             convex_hull, enclose, enclose_points, enlarge, empty, horzcat, ctranspose, diag,
//             dirac, infimum, interval_matrix, is_bounded, is_full_dim, is_intersecting, isequal,
//             kron, less_equal, less, lift, log, max, min, minkowski_difference, minus,
//             matrix_power, divide, norm, unite, origin (one file each); rad (below); the
//             constructor checks its input (interval.cpp)
// See also:   cora/cont_set/cont_set.h, cora/cont_set/zonotope/zonotope.h
//
// CORA: contSet/@interval/interval.m

#pragma once

#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/set.h"
#include "cora/global/within_tol.h"

#include <limits>
#include <optional>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

class Interval : public ContSet {
  public:
    static constexpr ClassId kClassId = ClassId::interval;

    /// The interval of the bounds inf <= sup: columns (..., n, 1), or matrices (..., n, k) for an
    /// interval matrix.
    Interval(Tensor inf, Tensor sup);

    /// A random box in n dimensions (CORA: center U[-2, 2], radius R/2*u with R ~ U[0, 10],
    /// u ~ U[0, 1]).
    static Interval generate_random(int64_t n, Rng &rng);

    /// The lower and the upper bounds.
    const Tensor &inf() const noexcept { return inf_; }
    const Tensor &sup() const noexcept { return sup_; }

    /// Whether the interval is empty: bounds without columns, as CORA's interval.empty.
    bool is_empty() const { return inf_.shape().back() == 0; }

    /// (sup - inf) / 2, elementwise, of the shape of the bounds.
    Tensor rad() const { return (sup_ - inf_) * 0.5; }

    /// M * I enclosed by its interval hull: the center maps by M, the radius by |M|.
    Interval linear_map(const Tensor &M) const;

    /// The Minkowski sum: the bounds add.
    Interval plus(const Interval &I2) const;

    /// A batch of intervals of one dimension (and batch shape) as one.
    static Interval stack(const std::vector<Interval> &Is);

    /// Whether the point p (..., n, 1) is in the box enlarged by tol in every coordinate (absolute,
    /// or relative to the bound, as within_tol); for a batch, whether all points are.
    bool contains(const Tensor &p, double tol = 100 * std::numeric_limits<double>::epsilon()) const;

    // Construction and combination ----------------------------------------------------------------

    /// The box (-Inf, Inf)^n of n dimensions (CORA: interval.Inf).
    static Interval unbounded(int64_t n);

    /// The image of the elementwise absolute value: [max(0, inf, -sup), max(-inf, sup)].
    Interval abs() const;

    /// The intersection with I2 (CORA's and): empty (bounds with no columns) if the boxes are
    /// disjoint by more than eps.
    Interval intersect(const Interval &I2) const;

    /// The Cartesian product: the columns stacked, or for rows the rows joined side by side; an
    /// empty operand returns the other. A number (a point) p is the box [p, p].
    Interval cart_prod(const Interval &I2) const;
    Interval cart_prod(const Tensor &p) const;
    static Interval cart_prod(const Tensor &p, const Interval &I);

    /// The intervals joined along dim (1: stacked, 2: side by side), as CORA's cat.
    static Interval cat(int dim, const std::vector<Interval> &Is);

    /// The intervals joined side by side (CORA's horzcat; a number is a point interval).
    static Interval horzcat(const std::vector<Interval> &Is);

    /// The empty interval of n dimensions: bounds (n, 0).
    static Interval empty(int64_t n);

    /// The smallest box around the points (..., n, N): the minimum and maximum over the columns.
    static Interval enclose_points(const Tensor &points);

    /// The hull of the two boxes (CORA's convHull_, which ends in or); an empty operand returns
    /// the other.
    Interval convex_hull(const Interval &I2) const;

    /// CORA's enclose: the convex hull of the two boxes.
    Interval enclose(const Interval &I2) const;

    /// The box scaled by factor about its center: [c - r f, c + r f]. factor is a number or a
    /// column (..., n, 1), nonnegative. An unbounded half-width stays unbounded for factor > 1 and
    /// is refused (Error::not_supported) for factor < 1.
    Interval enlarge(const Tensor &factor) const;
    Interval enlarge(double factor) const;

    /// The matrix of bounds transposed (CORA's ctranspose, real).
    Interval ctranspose() const;

    /// A column (or row) as the diagonal matrix with it on the k-th diagonal, a matrix as the
    /// column of its k-th diagonal (CORA's diag(I, k)); off the diagonal the matrix is [0, 0].
    Interval diag(int64_t k = 0) const;

    /// The derivative of order n of the Dirac delta on the box, elementwise (CORA's dirac(n, I)):
    /// [0, 0] where 0 is not in the box, else infinite bounds as CORA fixes them.
    Interval dirac(int64_t n = 0) const;

    /// The lower bound (CORA's infimum).
    Tensor infimum() const;

    /// The interval matrix [c - r, c + r] of the center c and the radius r (CORA's
    /// intervalMatrix(center(I), rad(I))); an Interval of matrix bounds is CORA's intervalMatrix.
    Interval interval_matrix() const;

    /// Whether no bound is infinite (an empty interval is bounded); for a batch, whether all are.
    bool is_bounded() const;

    // Comparison and predicates -------------------------------------------------------------------

    /// Whether no coordinate has a width within 1e-8 of zero (CORA's withinTol); false for an
    /// empty interval; for a batch, whether all members are.
    bool is_full_dim() const;

    /// Whether the boxes meet (CORA's isIntersecting_): per coordinate the bounds overlap, up to
    /// tol as within_tol; false if one is empty. With a point p, whether p is in the box. For a
    /// batch, whether all members do.
    bool is_intersecting(const Interval &I2, double tol = kWithinTol) const;
    bool is_intersecting(const Tensor &p, double tol = kWithinTol) const;

    /// Whether the boxes have the same bounds, up to tol as within_tol (CORA's default eps);
    /// infinite bounds must agree exactly. Boxes of different shape are not equal. A point p is
    /// the box [p, p].
    bool isequal(const Interval &I2, double tol = std::numeric_limits<double>::epsilon()) const;
    bool isequal(const Tensor &p, double tol = std::numeric_limits<double>::epsilon()) const;

    /// Whether this box lies in I2 (CORA's le, I <= I2): inf2 <= inf and sup <= sup2 everywhere.
    bool less_equal(const Interval &I2) const;

    /// Whether this box lies strictly in I2 (CORA's lt, I < I2): inf2 < inf and sup < sup2.
    bool less(const Interval &I2) const;

    // Arithmetic ----------------------------------------------------------------------------------

    /// The difference of two boxes, [inf - sup2, sup - inf2]; with a column v, the box shifted by
    /// -v, and v - I the box [v - sup, v - inf] (CORA's minus).
    Interval minus(const Interval &I2) const;
    Interval minus(const Tensor &v) const;
    static Interval minus(const Tensor &v, const Interval &I);

    /// The box divided by a number d: the hull of the quotients of the bounds. d = 0 gives
    /// (-Inf, Inf); Error::out_of_domain if a quotient is NaN (CORA's mrdivide).
    Interval divide(double d) const;

    /// The Kronecker product, as the hull of the products at the corners (CORA's kron).
    Interval kron(const Interval &I2) const;
    Interval kron(const Tensor &B) const;
    static Interval kron(const Tensor &A, const Interval &I);

    /// The k-th power of a square interval matrix (CORA's mpower): the identity for k = 0, else
    /// the product A (A (... A)) of k factors, each entry the sum of the corner hulls. A 1 x 1
    /// base is Error::not_supported (CORA's power).
    Interval matrix_power(int64_t exponent) const;

    /// The elementwise larger (smaller) of two boxes: [max(inf, inf2), max(sup, sup2)] (CORA's
    /// max and min). Y is a box or any set (its interval hull), or a number or array of the
    /// shape of the bounds. Without an argument, the upper (lower) bound.
    Interval max(const ContSet &Y) const;
    Interval max(const Tensor &Y) const;
    Interval min(const ContSet &Y) const;
    Interval min(const Tensor &Y) const;
    Tensor max() const;
    Tensor min() const;

    /// The Minkowski difference {x | x + S in I}: [inf - inf2, sup - sup2] for a box S, a shift
    /// for a point S, and for another set the support function in the coordinate directions.
    /// An empty difference (a bound would cross) is the empty interval. type "exact" is refused
    /// for another set (Error::no_exact_alg); "inner" and the default compute the same.
    Interval minkowski_difference(const Interval &S) const;
    Interval minkowski_difference(const Tensor &p) const;
    Interval minkowski_difference(const ContSet &S,
                                  const std::optional<std::string> &type = std::nullopt) const;

    /// The smallest box around this and I2 (CORA's or); an empty operand returns the other. With
    /// a point p, the box grown to contain p.
    Interval unite(const Interval &I2) const;
    Interval unite(const Tensor &p) const;

    /// The norm of the largest member in absolute value: for type two, the 2-norm of
    /// max(|inf|, |sup|) (the spectral norm for a matrix); else the norm of |center| + rad
    /// (one, inf, fro). -Inf for an empty interval. Returns (..., 1, 1).
    Tensor norm(Tensor::Norm type = Tensor::Norm::two) const;

    // Embedding -----------------------------------------------------------------------------------

    /// The box in n dimensions with coordinate dims[j] (0-based) the j-th coordinate of this box
    /// and (-Inf, Inf) elsewhere (CORA's lift_); a repeated entry of dims keeps the last. An
    /// empty interval is Error::not_supported.
    Interval lift(int64_t N, const std::vector<int64_t> &dims) const;

    /// The origin {0} in n dimensions, n >= 1 (CORA's interval.origin).
    static Interval origin(int64_t n);

    /// The image under log, elementwise: [log(inf), log(sup)]; log(0) is -Inf, and a negative
    /// bound is Error::out_of_domain.
    Interval log() const;

    // Points and functions ------------------------------------------------------------------------

    /// The point where the ray start + s dir, s >= 0, leaves the box; start defaults to the
    /// center. An unbounded ray ends at +-Inf in the coordinates of dir.
    Tensor boundary_point(const Tensor &dir,
                          const std::optional<Tensor> &start = std::nullopt) const;

    /// The image under the function, elementwise: an interval hull of the images of the bounds.
    /// acos, asin, atanh need the bounds in [-1, 1] and acosh at least 1, else
    /// Error::out_of_domain.
    Interval acos() const;
    Interval asin() const;
    Interval atan() const;
    Interval acosh() const;
    Interval asinh() const;
    Interval atanh() const;

    /// The image under cos (CORA's case analysis of the multiples of 2 pi, not the interval
    /// arithmetic of cos at the bounds), cosh (three cases around 0) and exp, elementwise.
    Interval cos() const;
    Interval cosh() const;
    Interval exp() const;

    /// The hull of atan2(y, x) at the four corners, y from this interval and x from I2, as CORA
    /// (not an enclosure where the box meets the branch cut of atan2).
    Interval atan2(const Interval &I2) const;

  protected:
    int64_t dim_impl() const override { return inf_.shape()[inf_.shape().size() - 2]; }
    Tensor center_impl() const override;
    Tensor support_function_impl(const Tensor &d) const override;
    Interval interval_impl() const override { return *this; }
    Tensor rand_point_impl(int64_t N, Rng &rng) const override;
    Set project_impl(const std::vector<int64_t> &dims) const override;
    std::vector<Polygon> vertices_impl() const override;
    std::vector<std::vector<Point3>> vertices_3d_impl() const override;
    std::string display_impl() const override;

  private:
    Tensor inf_, sup_;
};

// Operators, as CORA writes them: intervals add, a column v translates, a number scales (a
// negative one swaps the bounds), and a matrix M * I is the interval hull of the image.
inline Interval operator+(const Interval &I, const Interval &I2) { return I.plus(I2); }
inline Interval operator+(const Interval &I, const Tensor &v) { return {I.inf() + v, I.sup() + v}; }
inline Interval operator+(const Tensor &v, const Interval &I) { return I + v; }
inline Interval operator-(const Interval &I, const Interval &I2) { return I.minus(I2); }
inline Interval operator-(const Interval &I, const Tensor &v) { return I.minus(v); }
inline Interval operator-(const Tensor &v, const Interval &I) { return Interval::minus(v, I); }
inline Interval operator/(const Interval &I, double d) { return I.divide(d); }
inline bool operator<=(const Interval &I, const Interval &I2) { return I.less_equal(I2); }
inline bool operator<(const Interval &I, const Interval &I2) { return I.less(I2); }
inline Interval operator*(double s, const Interval &I) {
    return s >= 0 ? Interval(I.inf() * s, I.sup() * s) : Interval(I.sup() * s, I.inf() * s);
}
inline Interval operator*(const Interval &I, double s) { return s * I; }
inline Interval operator-(const Interval &I) { return -1.0 * I; }
inline Interval operator*(const Tensor &M, const Interval &I) { return I.linear_map(M); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
