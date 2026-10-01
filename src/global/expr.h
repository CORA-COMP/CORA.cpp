// expr - symbolic expressions of the state, the dynamics of a nonlinear system
//
// A dynamics function is written once on Expr and then differentiated symbolically, as CORA
// derives the Jacobian and the Hessian of a nonlinearSys, e.g. the van der Pol oscillator
// f(x) = {x[1], (1 - x[0]*x[0])*x[1] - x[0]}.
// An expression is evaluated on tensors (elementwise, so one call handles a whole batch of
// points), or over a box (an interval enclosure of its range); both stay differentiable.
//
// Syntax:     Expr x = Expr::var(0);   Expr f = sin(x) * x + 1;   f.diff(0).eval({0.5});
// Operations: + - * / (also with numbers), pow(e, n) for integer n, sin, cos, exp; diff, eval,
//             evalTensor, enclose. Range is the interval of tensors (rounding is not directed).
// See also:   contDynamics/nonlinearSys/nonlinearSys.h

#pragma once

#include "global/tensor/tensor.h"

#include <memory>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The interval [lo, hi] of every element of two tensors of one shape.
struct Range {
    Tensor lo, hi;

    Range() = default;
    Range(Tensor lo, Tensor hi) : lo(std::move(lo)), hi(std::move(hi)) {}
    /// The point range [p, p].
    explicit Range(const Tensor &p) : lo(p), hi(p) {}

    Tensor mid() const { return (lo + hi) * 0.5; }
    Tensor rad() const { return (hi - lo) * 0.5; }
};

Range operator+(const Range &a, const Range &b);
Range operator-(const Range &a, const Range &b);
Range operator*(const Range &a, const Range &b);
Range operator*(const Range &a, double s);

/// The square of a range: one that contains 0 squares to [0, ...], not to [-..., ...].
Range square(const Range &a);

class Expr {
  public:
    /// A constant.
    Expr(double value);

    /// The i-th state variable, i = 0, 1, ...
    static Expr var(int i);

    /// The derivative with respect to the i-th variable.
    Expr diff(int i) const;

    /// Whether the expression is the constant 0.
    bool isZero() const;

    /// The value at x, one tensor per variable; all of one shape, and so is the result.
    Tensor evalTensor(const std::vector<Tensor> &x) const;

    /// The value at the point x (one number per variable), computed on the current backend.
    double eval(const std::vector<double> &x) const;

    /// A range that contains every value of the expression over the box x (one Range per
    /// variable, all of one shape); it is exact for a variable that occurs once.
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

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
