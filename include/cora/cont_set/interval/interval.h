// interval - the class of intervals I = {x | inf <= x <= sup}, as CORA's interval
//
// A box with column bounds (..., n, 1). As in CORA the bounds may be matrices (..., n, n): an
// interval matrix is an Interval too, and Zonotope::linear_map takes it. Immutable: every
// operation returns a new interval.
//
// Syntax:     Interval I(inf, sup);   Interval I = Interval::generate_random(n, rng);
// Operators:  I + I2, I + v, I - v, -I, s * I, M * I, as in CORA
// Operations: support_function, linear_map, plus, rand_point, contains, stack, project, vertices,
//             generate_random, display, center, abs, intersect, cart_prod, cat, unbounded,
//             boundary_point, acos, asin, atan, atan2, acosh, asinh, atanh (one file each); rad
//             (below); the constructor checks its input (interval.cpp)
// See also:   cora/cont_set/cont_set.h, cora/cont_set/zonotope/zonotope.h
//
// CORA: contSet/@interval/interval.m

#pragma once

#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/set.h"

#include <limits>
#include <optional>
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
inline Interval operator-(const Interval &I, const Tensor &v) { return {I.inf() - v, I.sup() - v}; }
inline Interval operator*(double s, const Interval &I) {
    return s >= 0 ? Interval(I.inf() * s, I.sup() * s) : Interval(I.sup() * s, I.inf() * s);
}
inline Interval operator*(const Interval &I, double s) { return s * I; }
inline Interval operator-(const Interval &I) { return -1.0 * I; }
inline Interval operator*(const Tensor &M, const Interval &I) { return I.linear_map(M); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
