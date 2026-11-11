// taylm - the class of Taylor models, as CORA's taylm (minimal: what range bounding needs)
//
// A scalar T = p(a) + [rl, ru] over the variables a_1..a_k in [-1, 1], where p is a polynomial
// of degree at most max_order (CORA: 6) and the interval absorbs the rest. A box [lo, hi] is
// the model mid + rad a_i, a zonotope c + G a (one variable per generator). The sums, products
// and integer powers of models are enclosures; the range of a model is CORA's method "int":
// every monomial is bounded by [0, 1] (even exponents) or [-1, 1] and the bounds add up. An
// expression is evaluated on models with Taylm::evaluate: sums, products and integer powers stay
// polynomial, every other operation is replaced by the interval of its operand's range (sound,
// but the dependency on the variables is lost there). Not ported: the optimisation methods bnb,
// bernstein and linQuad, sin/cos/exp/log expansions, plot, subsref and the matrix functions.
//
// Syntax:     auto x = Taylm::from_interval(I);   Taylm f = Taylm::evaluate(expr, x);
//             auto [lo, hi] = f.bounds();   Interval r = f.interval();
// Operations: plus, minus, uminus, times, power, interval, evaluate (one file each in
//             src/cont_set/taylm/); the factories check their input (taylm.cpp)
// See also:   cora/global/expr.h, cora/cont_set/interval/interval.h
//
// CORA: contSet/@taylm/taylm.m
// CORA: contSet/@taylm/dim.m
// CORA authors: Dmitry Grebenyuk, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/expr.h"

#include <array>
#include <map>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

class Taylm {
  public:
    /// The exponents of the variables of one monomial.
    using Monomial = std::vector<int>;

    /// CORA's default for the degree of the polynomial part.
    static constexpr int kDefaultMaxOrder = 6;

    /// The constant c in `vars` variables.
    static Taylm constant(double c, int64_t vars, int max_order = kDefaultMaxOrder);

    /// A model with no dependency on the variables that holds the box [lo, hi]: the midpoint as
    /// the constant, the half-width as the remainder.
    static Taylm from_range(double lo, double hi, int64_t vars, int max_order = kDefaultMaxOrder);

    /// The models x_i = mid + rad a_i of the coordinates of the box (finite bounds, one variable
    /// per coordinate), CORA's taylm(interval).
    static std::vector<Taylm> from_interval(const Interval &I, int max_order = kDefaultMaxOrder);

    /// The models c_i + G(i, :) a of the coordinates of the zonotope (one variable per
    /// generator), CORA's taylm(zonotope).
    static std::vector<Taylm> from_zonotope(const Zonotope &Z, int max_order = kDefaultMaxOrder);

    /// The value of the expression f on the models x (one per variable of f): an enclosure of
    /// its range over the variables' box, where the dependency stays as far as f is polynomial.
    static Taylm evaluate(const Expr &f, const std::vector<Taylm> &x);

    int64_t num_vars() const noexcept { return vars_; }
    int max_order() const noexcept { return max_order_; }
    /// The polynomial part: the coefficient of every monomial that has one.
    const std::map<Monomial, double> &coefficients() const noexcept { return terms_; }
    /// The remainder [lo, hi] that the polynomial part leaves.
    std::array<double, 2> remainder() const noexcept { return {rem_lo_, rem_hi_}; }

    /// The bounds of the model over its variables (CORA's interval with method "int").
    std::array<double, 2> bounds() const;
    /// The same as a box of one dimension.
    Interval interval() const;

    // Arithmetic ----------------------------------------------------------------------------------

    /// The sum and the difference with a model of the same variables, and with a number.
    Taylm plus(const Taylm &S) const;
    Taylm plus(double s) const;
    Taylm minus(const Taylm &S) const;
    Taylm minus(double s) const;
    /// The negated model (CORA's uminus) and the model itself (CORA's uplus).
    Taylm uminus() const;
    Taylm uplus() const { return *this; }
    /// The product with a model of the same variables, and with a number: the terms above
    /// max_order and the remainders go to the remainder.
    Taylm times(const Taylm &S) const;
    Taylm times(double s) const;
    /// The n-th power for an integer n >= 0 (CORA's power/mpower), by repeated products.
    Taylm power(int n) const;

  private:
    Taylm(int64_t vars, int max_order) : vars_(vars), max_order_(max_order) {}
    /// The range of the polynomial part alone, by the monomial bounds.
    std::array<double, 2> polynomial_bounds() const;

    int64_t vars_;
    int max_order_;
    std::map<Monomial, double> terms_;
    double rem_lo_ = 0, rem_hi_ = 0;
};

inline Taylm operator+(const Taylm &a, const Taylm &b) { return a.plus(b); }
inline Taylm operator-(const Taylm &a, const Taylm &b) { return a.minus(b); }
inline Taylm operator-(const Taylm &a) { return a.uminus(); }
inline Taylm operator*(const Taylm &a, const Taylm &b) { return a.times(b); }
inline Taylm operator*(double s, const Taylm &a) { return a.times(s); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
