// interval - the class of intervals I = {x | inf <= x <= sup}, as CORA's interval
//
// A box with column bounds (..., n, 1). As in CORA the bounds may be matrices (..., n, n): an
// interval matrix is an Interval too, and Zonotope::linear_map takes it. Immutable: every
// operation returns a new interval.
//
// Syntax:     Interval I(inf, sup);   Interval I = Interval::generate_random(rng);
// Operators:  I + I2, I + v, I - I2, I - v, v - I, -I, +I, s * I, M * I, I / s, I <= I2, I < I2, as
//             in CORA
// Operations: support_function, linear_map, plus, rand_point, contains, stack, project, vertices,
//             generate_random, display, center, abs, intersect, cart_prod, cat, unbounded,
//             boundary_point, acos, asin, atan, atan2, acosh, asinh, atanh, cos, cosh, exp,
//             convex_hull, enclose, enclose_points, enlarge, empty, horzcat, ctranspose, diag,
//             dirac, infimum, interval_matrix, is_bounded, is_full_dim, is_intersecting, isequal,
//             kron, less_equal, less, lift, log, max, min, minkowski_difference, minus,
//             matrix_power, divide, norm, unite, origin, rad, radius, sign, round, sin, sinh,
//             power, rdivide, prod, partition, project_high_dim, quad_map, reduce, representsa,
//             split, sqrt, sum, supremum, tan, tanh, times, transpose, tril, triu, uminus, uplus,
//             vertcat, volume, zonotope (one file each); the constructor checks its input
//             (interval.cpp)
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
#include <utility>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

class Zonotope;

class Interval : public ContSet {
  public:
    static constexpr ClassId kClassId = ClassId::interval;

    /// The interval of the bounds inf <= sup: columns (..., n, 1), or matrices (..., n, k) for an
    /// interval matrix.
    Interval(Tensor inf, Tensor sup);

    /// A random box, as CORA's generateRandom: dimension (default that of center or max_radius,
    /// else random in 1..10), center (default U[-2, 2] per entry), max_radius (radii >= 0: a vector,
    /// or a 1 x 1 tensor for all; default U[0, 10] per entry); the radius is max_radius/2*u,
    /// u ~ U[0, 1].
    static Interval generate_random(Rng &rng, std::optional<int64_t> dimension = std::nullopt,
                                    const std::optional<Tensor> &center = std::nullopt,
                                    const std::optional<Tensor> &max_radius = std::nullopt);

    /// The lower and the upper bounds.
    const Tensor &inf() const noexcept { return inf_; }
    const Tensor &sup() const noexcept { return sup_; }

    /// Whether the interval is empty: bounds without columns, as CORA's interval.empty.
    bool is_empty() const { return inf_.shape().back() == 0; }

    /// (sup - inf) / 2, elementwise, of the shape of the bounds.
    Tensor rad() const;

    /// M * I enclosed by its interval hull: the center maps by M, the radius by |M|.
    Interval linear_map(const Tensor &M) const;

    /// The Minkowski sum: the bounds add.
    Interval plus(const Interval &I2) const;

    /// A batch of intervals of one dimension (and batch shape) as one.
    static Interval stack(const std::vector<Interval> &Is);

    /// Whether the point p (..., n, 1) is in the box enlarged by tol in every coordinate (absolute,
    /// or relative to the bound, as within_tol); for a batch, whether all points are. tol
    /// default: the repository tolerance (docs/architecture.md).
    bool contains(const Tensor &p, Tol tol = std::nullopt) const;

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

    /// The intervals stacked (CORA's vertcat; a number is a point interval); they agree in the
    /// columns.
    static Interval vertcat(const std::vector<Interval> &Is);

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

    /// The matrix of bounds transposed (CORA's transpose; the same as ctranspose for real bounds).
    Interval transpose() const;

    /// A column (or row) as the diagonal matrix with it on the k-th diagonal, a matrix as the
    /// column of its k-th diagonal (CORA's diag(I, k)); off the diagonal the matrix is [0, 0].
    Interval diag(int64_t k = 0) const;

    /// The derivative of order n of the Dirac delta on the box, elementwise (CORA's dirac(n, I)):
    /// [0, 0] where 0 is not in the box, else infinite bounds as CORA fixes them.
    Interval dirac(int64_t n = 0) const;

    /// The lower bound (CORA's infimum).
    Tensor infimum() const;

    /// The upper bound (CORA's supremum).
    Tensor supremum() const;

    /// The interval matrix [c - r, c + r] of the center c and the radius r (CORA's
    /// intervalMatrix(center(I), rad(I))); an Interval of matrix bounds is CORA's intervalMatrix.
    Interval interval_matrix() const;

    /// Whether no bound is infinite (an empty interval is bounded); for a batch, whether all are.
    bool is_bounded() const;

    // Comparison and predicates -------------------------------------------------------------------

    /// Whether no coordinate has a width within tol of zero (as within_tol); false for an empty
    /// interval; for a batch, whether all members are. tol default: the repository tolerance.
    bool is_full_dim(Tol tol = std::nullopt) const;

    /// Whether the boxes meet (CORA's isIntersecting_): per coordinate the bounds overlap, up to
    /// tol as within_tol; false if one is empty. With a point p, whether p is in the box. For a
    /// batch, whether all members do. tol default: the repository tolerance.
    bool is_intersecting(const Interval &I2, Tol tol = std::nullopt) const;
    bool is_intersecting(const Tensor &p, Tol tol = std::nullopt) const;

    /// Whether the boxes have the same bounds, up to tol as within_tol; infinite bounds must
    /// agree exactly. Boxes of different shape are not equal. A point p is the box [p, p]. tol
    /// default: the repository tolerance.
    bool isequal(const Interval &I2, Tol tol = std::nullopt) const;
    bool isequal(const Tensor &p, Tol tol = std::nullopt) const;

    /// Whether this box lies in I2 (CORA's le, I <= I2): inf2 <= inf and sup <= sup2 everywhere.
    bool less_equal(const Interval &I2) const;

    /// Whether this box lies strictly in I2 (CORA's lt, I < I2): inf2 < inf and sup < sup2.
    bool less(const Interval &I2) const;

    // Arithmetic ----------------------------------------------------------------------------------

    /// The negated box [-sup, -inf] (CORA's uminus) and the box itself (CORA's uplus).
    Interval uminus() const;
    Interval uplus() const;

    /// The entrywise product with a box or with numbers (CORA's times, I .* F): the hull of the
    /// corner products, a corner 0 * Inf taken as 0; the operands broadcast (a 1 x 1 factor is a
    /// scalar).
    Interval times(const Interval &I2) const;
    Interval times(const Tensor &factor) const;

    /// The sum of the entries along dim (1: down the rows, a row result; 2: along the columns, a
    /// column result); default: 1, or 2 for a single row (CORA's sum). The bounds add.
    Interval sum(std::optional<int> dim = std::nullopt) const;

    /// The volume of the box, (..., 1, 1): the product of the widths, 0 for an empty or a
    /// degenerate box (CORA's volume_).
    Tensor volume() const;

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

    /// The 2-norm of the radius over all entries, (..., 1, 1) (CORA's radius).
    Tensor radius() const;

    // Powers, quotients and products --------------------------------------------------------------

    /// The elementwise power (CORA's power): the exponent is a number, numbers of the shape of the
    /// bounds (or any shape when the box is one number), or a box; the base may be numbers and the
    /// exponent a box. Integer exponents take the hull of the end powers (0 below an even power
    /// over a sign change), negative ones the power of the reciprocal; the others are
    /// [inf^e, sup^e]. Error::out_of_domain if a bound is NaN.
    Interval power(double exponent) const;
    Interval power(const Tensor &exponent) const;
    Interval power(const Interval &exponent) const;
    static Interval power(const Tensor &base, const Interval &exponent);

    /// The entrywise quotient (CORA's rdivide): a box over numbers (the hull of the quotients,
    /// the whole line where the number is 0), numbers over a box (the hull, the whole line over a
    /// box with 0 inside, the ray past n/inf at sup = 0), and a box over a box (I .* (1 ./ D)).
    /// Error::out_of_domain over [0, 0] or if a quotient is NaN.
    Interval rdivide(const Tensor &denominator) const;
    Interval rdivide(const Interval &denominator) const;
    static Interval rdivide(const Tensor &numerator, const Interval &denominator);

    /// The product of the rows (dim 1, a row result) or the columns (dim 2, a column) of an
    /// interval matrix, folded from the last factor (CORA's prod); default dim: 1, or 2 for a row.
    Interval prod(std::optional<int> dim = std::nullopt) const;

    // Embedding and cutting -----------------------------------------------------------------------

    /// The boxes of the equal grid cut: splits per coordinate, splits^n boxes in the order of
    /// nested loops (the first coordinate slowest), neighbours touching or overlapping (CORA's
    /// partition).
    std::vector<Interval> partition(int64_t splits) const;

    /// The box in N dimensions with coordinate proj[j] (0-based) the j-th coordinate of this box
    /// and [0, 0] elsewhere (CORA's projectHighDim_); the empty interval stays empty.
    Interval project_high_dim(int64_t N, const std::vector<int64_t> &proj) const;

    /// The box itself (CORA's reduce of an interval); order must be at least 1.
    Interval reduce(int order = 1) const;

    /// The two halves of the box cut at the center of coordinate n (0-based): the lower half has
    /// the center as upper bound there, the upper half as lower bound (CORA's split).
    std::pair<Interval, Interval> split(int64_t n) const;

    /// The lower (upper) triangular part of an interval matrix from the k-th diagonal on, k >= 0;
    /// the other entries are [0, 0] (CORA's tril and triu).
    Interval tril(int64_t K = 0) const;
    Interval triu(int64_t K = 0) const;

    /// The zonotope with the center c and the generators diag(r), r the radius reaching both ends
    /// from c (CORA's zonotope); the columns of zero radius are dropped (those of zero radius in
    /// every member of a batch).
    Zonotope zonotope() const;

    // Rounding and signs --------------------------------------------------------------------------

    /// The bounds rounded to digits decimals, halves away from zero (CORA's round); type
    /// "significant" is refused (Error::not_supported).
    Interval round(int64_t digits = 0, const std::string &type = "decimals") const;

    /// The hull of the sign of the bounds: [sign(inf), sign(sup)].
    Interval sign() const;

    /// Whether the box is also a set of the CORA type (CORA's representsa_): "origin", "point",
    /// "emptySet", "fullspace", "interval", "zonotope", "capsule", ... ; for a batch whether all
    /// members are. tol default: the repository tolerance. Not a returned set: convert instead.
    bool representsa(const std::string &type, Tol tol = std::nullopt) const;

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

    /// The image under sin, as CORA computes it: cos(I - pi/2) (the case analysis of cos), and
    /// sinh: [sinh(inf), sinh(sup)].
    Interval sin() const;
    Interval sinh() const;

    /// The box around the image of the quadratic map x -> (x' Q_i x)_i over this box (CORA's
    /// quadMap): the box of Zonotope::quad_map of its zonotope. With I2, x1' Q_i x2 for x1 in
    /// this box and x2 in I2. Q[i] are the (n, n) matrices of the outputs; the boxes must be
    /// bounded.
    Interval quad_map(const std::vector<Tensor> &Q) const;
    Interval quad_map(const Interval &I2, const std::vector<Tensor> &Q) const;

    /// The image under sqrt and tanh: [f(inf), f(sup)] (both increase); sqrt needs inf >= 0, else
    /// Error::out_of_domain.
    Interval sqrt() const;
    Interval tanh() const;

    /// The image under tan: [tan(inf), tan(sup)] where the box is narrower than pi and does not
    /// reach past a pole, else (-Inf, Inf), as CORA.
    Interval tan() const;

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

    // The generic operations of ContSet, which a box answers by its own (defined beside them).
    bool is_bounded_impl() const override;
    bool is_full_dim_impl(Tol tol) const override;
    Set reduce_impl(int order) const override;
    Set linear_map_impl(const Tensor &M) const override;
    Set translate_impl(const Tensor &v) const override;
    Set enlarge_impl(const Tensor &factor) const override;
    Set uminus_impl() const override;
    Set compact_impl(const std::optional<std::string> &method, Tol tol) const override;
    Set uplus_impl() const override;
    Set times_impl(const Tensor &factor) const override;
    Set minkowski_difference_impl(const ContSet &S2,
                                  const std::optional<std::string> &type) const override;
    Set project_high_dim_impl(int64_t N, const std::vector<int64_t> &proj) const override;
    Set lift_impl(int64_t N, const std::vector<int64_t> &proj) const override;
    Tensor norm_impl(Tensor::Norm type, const std::string &mode) const override;
    Set enclose_impl(const ContSet &S2) const override;

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
inline Interval operator-(const Interval &I) { return I.uminus(); }
inline Interval operator+(const Interval &I) { return I.uplus(); }
inline Interval operator*(const Tensor &M, const Interval &I) { return I.linear_map(M); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
