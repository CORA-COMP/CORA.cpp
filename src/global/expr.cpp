// expr - symbolic expressions of the state, the dynamics of a nonlinear system
//
// The trees of expressions, their derivative, and their evaluation at points and over boxes.
//
// Syntax:   see expr.h
// See also: contDynamics/nonlinearSys/nonlinearSys.h

#include "global/expr.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::ct {

enum class Op { Const, Var, Add, Sub, Mul, Div, Neg, Pow, Sin, Cos, Exp };

struct Expr::Node {
    Op op;
    double value = 0;  // Const
    int index = 0;     // Var: the variable; Pow: the exponent
    Expr a, b;         // the operands
};

/// Builds the nodes, and folds constants so that derivatives stay small.
struct ExprMaker {
    static Expr make(Op op, Expr a = Expr(), Expr b = Expr(), double value = 0, int index = 0) {
        return Expr(std::make_shared<const Expr::Node>(Expr::Node{op, value, index, a, b}));
    }
    static const Expr::Node &node(const Expr &e) { return *e.node_; }
    static bool isConst(const Expr &e, double v) {
        return e.node_->op == Op::Const && e.node_->value == v;
    }
    static bool isConst(const Expr &e) { return e.node_->op == Op::Const; }
};

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using M = ExprMaker;

const double kPi = 3.14159265358979323846;

/// Whether the interval [lo, hi] contains a point of the form offset + 2 pi k.
bool aux_hasPeak(double lo, double hi, double offset) {
    return std::ceil((lo - offset) / (2 * kPi)) <= std::floor((hi - offset) / (2 * kPi));
}

/// The range of sin (phase = 0) or cos (phase = pi/2, as cos x = sin(x + pi/2)) over [lo, hi].
Range aux_sinRange(double lo, double hi, double phase) {
    lo += phase;
    hi += phase;
    if (hi - lo >= 2 * kPi) return {-1, 1};
    Range r{std::min(std::sin(lo), std::sin(hi)), std::max(std::sin(lo), std::sin(hi))};
    if (aux_hasPeak(lo, hi, kPi / 2)) r.hi = 1;
    if (aux_hasPeak(lo, hi, -kPi / 2)) r.lo = -1;
    return r;
}

Range aux_div(const Range &a, const Range &b) {
    if (b.lo <= 0 && b.hi >= 0)
        throw std::domain_error("cora: an expression divides by a range that contains 0; "
                                "the box is too large for this dynamics");
    return a * Range(1 / b.hi, 1 / b.lo);
}

Range aux_pow(const Range &a, int n) {
    if (n < 0) return aux_div(Range(1), aux_pow(a, -n));
    if (n == 0) return Range(1);
    const double p = std::pow(a.lo, n), q = std::pow(a.hi, n);
    if (n % 2 == 1) return {p, q};  // odd powers grow
    if (a.lo <= 0 && a.hi >= 0) return {0, std::max(p, q)};  // an even power dips to 0
    return {std::min(p, q), std::max(p, q)};
}

/// The value of a node from the values of its operands, at a point.
double aux_apply(const Expr::Node &n, double a, double b) {
    switch (n.op) {
    case Op::Add: return a + b;
    case Op::Sub: return a - b;
    case Op::Mul: return a * b;
    case Op::Div: return a / b;
    case Op::Neg: return -a;
    case Op::Pow: return std::pow(a, n.index);
    case Op::Sin: return std::sin(a);
    case Op::Cos: return std::cos(a);
    case Op::Exp: return std::exp(a);
    default: break;
    }
    throw std::logic_error("cora: an expression node of an unknown operation");
}

/// The same over boxes.
Range aux_apply(const Expr::Node &n, const Range &a, const Range &b) {
    switch (n.op) {
    case Op::Add: return a + b;
    case Op::Sub: return a - b;
    case Op::Mul: return a * b;
    case Op::Div: return aux_div(a, b);
    case Op::Neg: return {-a.hi, -a.lo};
    case Op::Pow: return aux_pow(a, n.index);
    case Op::Sin: return aux_sinRange(a.lo, a.hi, 0);
    case Op::Cos: return aux_sinRange(a.lo, a.hi, kPi / 2);
    case Op::Exp: return {std::exp(a.lo), std::exp(a.hi)};
    default: break;
    }
    throw std::logic_error("cora: an expression node of an unknown operation");
}

/// The value of an expression over the point or box x.
template <class T> T aux_evaluate(const Expr &e, const std::vector<T> &x) {
    const Expr::Node &n = M::node(e);
    if (n.op == Op::Const) return T(n.value);
    if (n.op == Op::Var) {
        if (n.index >= static_cast<int>(x.size()))
            throw std::out_of_range("cora: the expression uses a variable the point does not have");
        return x[n.index];
    }
    const T a = aux_evaluate(n.a, x);
    const bool binary = n.op == Op::Add || n.op == Op::Sub || n.op == Op::Mul || n.op == Op::Div;
    return aux_apply(n, a, binary ? aux_evaluate(n.b, x) : T(0));
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Range ---------------------------------------------------------------------------------------

Range operator+(const Range &a, const Range &b) { return {a.lo + b.lo, a.hi + b.hi}; }

Range operator-(const Range &a, const Range &b) { return {a.lo - b.hi, a.hi - b.lo}; }

Range operator*(const Range &a, const Range &b) {
    const double p[4] = {a.lo * b.lo, a.lo * b.hi, a.hi * b.lo, a.hi * b.hi};
    return {*std::min_element(p, p + 4), *std::max_element(p, p + 4)};
}

// Building ------------------------------------------------------------------------------------

Expr::Expr(double value) : node_(std::make_shared<const Node>(Node{Op::Const, value, 0, {}, {}})) {}

Expr Expr::var(int i) {
    if (i < 0) throw std::invalid_argument("cora: the index of a variable is 0, 1, 2, ...");
    return M::make(Op::Var, Expr(), Expr(), 0, i);
}

// Every operation folds constants (0 + x, 1 * x, 2 * 3), which keeps derivatives small.
Expr operator+(const Expr &a, const Expr &b) {
    if (M::isConst(a, 0)) return b;
    if (M::isConst(b, 0)) return a;
    if (M::isConst(a) && M::isConst(b)) return Expr(M::node(a).value + M::node(b).value);
    return M::make(Op::Add, a, b);
}

Expr operator-(const Expr &a, const Expr &b) {
    if (M::isConst(b, 0)) return a;
    if (M::isConst(a) && M::isConst(b)) return Expr(M::node(a).value - M::node(b).value);
    return M::make(Op::Sub, a, b);
}

Expr operator*(const Expr &a, const Expr &b) {
    if (M::isConst(a, 0) || M::isConst(b, 0)) return Expr(0.0);
    if (M::isConst(a, 1)) return b;
    if (M::isConst(b, 1)) return a;
    if (M::isConst(a) && M::isConst(b)) return Expr(M::node(a).value * M::node(b).value);
    return M::make(Op::Mul, a, b);
}

// Dividing by the constant 0 is an error here; a range around 0 is one in enclose.
Expr operator/(const Expr &a, const Expr &b) {
    if (M::isConst(b, 0)) throw std::domain_error("cora: an expression divides by the constant 0");
    if (M::isConst(a, 0)) return Expr(0.0);
    if (M::isConst(b, 1)) return a;
    return M::make(Op::Div, a, b);
}

Expr operator-(const Expr &a) {
    if (M::isConst(a)) return Expr(-M::node(a).value);
    return M::make(Op::Neg, a);
}

Expr pow(const Expr &a, int n) {
    if (n == 0) return Expr(1.0);
    if (n == 1) return a;
    return M::make(Op::Pow, a, M::make(Op::Const), 0, n);
}

// The functions are plain nodes; diff knows their derivatives.
Expr sin(const Expr &a) { return M::make(Op::Sin, a); }
Expr cos(const Expr &a) { return M::make(Op::Cos, a); }
Expr exp(const Expr &a) { return M::make(Op::Exp, a); }

// Derivative and evaluation -------------------------------------------------------------------

Expr Expr::diff(int i) const {
    const Node &n = *node_;
    if (n.op == Op::Const) return Expr(0.0);
    if (n.op == Op::Var) return Expr(n.index == i ? 1.0 : 0.0);
    const Expr da = n.a.diff(i);  // the chain rule needs the inner derivative for every operation
    switch (n.op) {
    case Op::Add: return da + n.b.diff(i);
    case Op::Sub: return da - n.b.diff(i);
    case Op::Mul: return da * n.b + n.a * n.b.diff(i);
    case Op::Div: return (da * n.b - n.a * n.b.diff(i)) / (n.b * n.b);
    case Op::Neg: return -da;
    case Op::Pow: return Expr(double(n.index)) * pow(n.a, n.index - 1) * da;
    case Op::Sin: return cos(n.a) * da;
    case Op::Cos: return -(sin(n.a) * da);
    case Op::Exp: return exp(n.a) * da;
    default: break;
    }
    throw std::logic_error("cora: an expression node of an unknown operation");
}

double Expr::eval(const std::vector<double> &x) const { return aux_evaluate<double>(*this, x); }

Range Expr::enclose(const std::vector<Range> &x) const { return aux_evaluate<Range>(*this, x); }

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
