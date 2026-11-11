// expr - symbolic expressions of the state, the dynamics of a nonlinear system
//
// A dynamics function is written once on Expr and then differentiated symbolically, as CORA
// derives the Jacobian and the Hessian of a nonlinearSys, e.g. the van der Pol oscillator
// f(x) = {x[1], (1 - x[0]*x[0])*x[1] - x[0]}.
// An expression is evaluated on tensors (elementwise, so one call handles a whole batch of
// points), or over a box (an interval enclosure of its range); both stay differentiable.
//
// Syntax:     Expr x = Expr::var(0);   Expr f = sin(x) * x + 1;   f.diff(0).eval({0.5});
// Operations: + - * / (also with numbers), pow(e, n) for an integer or real n, sqrt, sin, cos,
//             tan, exp; diff, eval, eval_tensor, enclose (of one expression or of a list, which
//             shares the values of common subexpressions). Range is the interval of tensors
//             (rounding is not directed).
// See also:   cora/cont_dynamics/nonlinear_sys/nonlinear_sys.h
//
// Authors:       Tobias Ladner, Claude Opus 5.5, Claude Sonnet 5.5
// Written:       02-November-2026
// Last update:   11-November-2026
// Reviewed:      ---

#pragma once

#include "cora/global/tensor/tensor.h"

#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// Ranges ------------------------------------------------------------------------------------------

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

// Expressions -------------------------------------------------------------------------------------

/// The operation of a node of the tree (see Expr::view).
enum class ExprOp { constant, var, add, sub, mul, div, neg, pow, rpow, sin, cos, tan, exp };

struct ExprView;

class Expr {
  public:
    /// A constant.
    Expr(double value);

    /// The i-th state variable, i = 0, 1, ...
    static Expr var(int i);

    /// The derivative with respect to the i-th variable.
    Expr diff(int i) const;

    /// Whether the expression is the constant 0.
    bool is_zero() const;

    /// The value at x, one tensor per variable; all of one shape, and so is the result.
    Tensor eval_tensor(const std::vector<Tensor> &x) const;

    /// The value at the point x (one number per variable), computed on the current backend.
    double eval(const std::vector<double> &x) const;

    /// A range that contains every value of the expression over the box x (one Range per
    /// variable, all of one shape); it is exact for a variable that occurs once.
    Range enclose(const std::vector<Range> &x) const;

    /// The values of all expressions at x; a subexpression they share is evaluated once.
    static std::vector<Tensor> eval_tensor(const std::vector<Expr> &exprs,
                                           const std::vector<Tensor> &x);

    /// The tree as the JSON of the CORALean oracle: {"x":i} for a state, {"u":j} for an input (the
    /// variables from `states` on), {"const":text} with the text `number` gives,
    /// {"op":..,"args":..}. Throws for a power of a non-integer other than 1/2 (the square root).
    std::string to_json(int states, const std::function<std::string(double)> &number) const;

    /// The ranges of all expressions over the box x; a shared subexpression is enclosed once.
    static std::vector<Range> enclose(const std::vector<Expr> &exprs, const std::vector<Range> &x);

    /// The root of the tree for a walk over it: the operation, its constant (`value`: the number
    /// of a constant, the real exponent of an rpow), its index (the variable of a var, the integer
    /// exponent of a pow) and the operands `a` and `b` (empty where the operation has none).
    ExprView view() const;

    /// The identity of the node: two expressions that share a subtree have the same id there.
    const void *id() const noexcept { return node_.get(); }
    /// As enclose, one entry per expression, in one pass over the nonzero ones; a zero is empty.
    static std::vector<std::optional<Range>> enclose_nonzero(const std::vector<Expr> &exprs,
                                                             const std::vector<Range> &x);

    /// The (rows, cols) matrix of the values at x of the expressions from `first` on, row by row;
    /// a zero entry stays a zero without an evaluation. `like` gives the backend of the zeros.
    static Tensor eval_matrix(const std::vector<Expr> &exprs, std::size_t first, int64_t rows,
                              int64_t cols, const std::vector<Tensor> &x, const Tensor &like);

    struct Node; // the tree behind an expression

  private:
    Expr() = default; // an empty operand of a node
    explicit Expr(std::shared_ptr<const Node> node) : node_(std::move(node)) {}

    std::shared_ptr<const Node> node_;

    friend struct ExprMaker; // builds the nodes (expr.cpp)
};

/// One level of an expression tree.
struct ExprView {
    ExprOp op;
    double value;
    int index;
    Expr a, b;
};
/// The rows of X (n, N) as n tensors (N, 1), one per variable, as eval_tensor takes them.
std::vector<Tensor> variable_columns(const Tensor &X);

Expr operator+(const Expr &a, const Expr &b);
Expr operator-(const Expr &a, const Expr &b);
Expr operator*(const Expr &a, const Expr &b);
Expr operator/(const Expr &a, const Expr &b);
Expr operator-(const Expr &a);

/// a^n for an integer n.
Expr pow(const Expr &a, int n);
/// a^p for a real p (an integer p is the power above); a must be positive where it is evaluated.
Expr pow(const Expr &a, double p);
/// The square root, a^(1/2).
Expr sqrt(const Expr &a);
Expr sin(const Expr &a);
Expr cos(const Expr &a);
Expr tan(const Expr &a);
Expr exp(const Expr &a);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
