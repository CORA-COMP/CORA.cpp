// expr - symbolic expressions of the state, the dynamics of a nonlinear system
//
// A dynamics function is written once on Expr and then differentiated symbolically, as CORA
// derives the Jacobian and the Hessian of a nonlinearSys, e.g. the van der Pol oscillator
// f(x) = {x[1], (1 - x[0]*x[0])*x[1] - x[0]}.
// An expression is evaluated at a point, or over a box (an interval enclosure of its range).
//
// Syntax:     Expr x = Expr::var(0);   Expr f = sin(x) * x + 1;   f.diff(0).eval({0.5});
// Operations: + - * / (also with numbers), pow(e, n) for integer n, sin, cos, exp; diff, eval,
//             enclose. Range is the interval of one number (rounding is not directed).
// See also:   contDynamics/nonlinearSys/nonlinearSys.h

#pragma once

#include <memory>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::ct {

/// The interval [lo, hi] of one number.
struct Range {
    double lo = 0, hi = 0;

    Range() = default;
    Range(double value) : lo(value), hi(value) {}
    Range(double lo, double hi) : lo(lo), hi(hi) {}

    double mid() const { return 0.5 * (lo + hi); }
    double rad() const { return 0.5 * (hi - lo); }
};

Range operator+(const Range &a, const Range &b);
Range operator-(const Range &a, const Range &b);
Range operator*(const Range &a, const Range &b);

class Expr {
  public:
    /// A constant.
    Expr(double value);

    /// The i-th state variable, i = 0, 1, ...
    static Expr var(int i);

    /// The derivative with respect to the i-th variable.
    Expr diff(int i) const;

    /// The value at the point x (one number per variable).
    double eval(const std::vector<double> &x) const;

    /// An interval that contains every value of the expression over the box x (one Range per
    /// variable); it is exact for a variable that occurs once.
    Range enclose(const std::vector<Range> &x) const;

    struct Node;  // the tree behind an expression

  private:
    Expr() = default;  // an empty operand of a node
    explicit Expr(std::shared_ptr<const Node> node) : node_(std::move(node)) {}

    std::shared_ptr<const Node> node_;

    friend struct ExprMaker;  // builds the nodes (expr.cpp)
};

Expr operator+(const Expr &a, const Expr &b);
Expr operator-(const Expr &a, const Expr &b);
Expr operator*(const Expr &a, const Expr &b);
Expr operator/(const Expr &a, const Expr &b);
Expr operator-(const Expr &a);

/// a^n for an integer n.
Expr pow(const Expr &a, int n);
Expr sin(const Expr &a);
Expr cos(const Expr &a);
Expr exp(const Expr &a);

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
