// expr - symbolic expressions of the state, the dynamics of a nonlinear system
//
// The trees of expressions, their derivative, and their evaluation at tensors and over boxes.
//
// Syntax:   see expr.h
// See also: cora/cont_dynamics/nonlinear_sys/nonlinear_sys.h
//
// Authors:       Claude Sonnet 5.5, Tobias Ladner, Claude Opus 5.5
// Written:       29-September-2026
// Last update:   09-November-2026
// Reviewed:      ---

#include "cora/global/expr.h"
#include "cora/global/error.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <utility>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

enum class Op { constant, var, add, sub, mul, div, neg, pow, rpow, sin, cos, tan, exp };

struct Expr::Node {
    Op op;
    double value = 0; // Const: the value; Rpow: the real exponent
    int index = 0;    // Var: the variable; Pow: the integer exponent
    Expr a, b;        // the operands
};

/// Builds the nodes, and folds constants so that derivatives stay small.
struct ExprMaker {
    static Expr make(Op op, Expr a = Expr(), Expr b = Expr(), double value = 0, int index = 0) {
        return Expr(std::make_shared<const Expr::Node>(
            Expr::Node{op, value, index, std::move(a), std::move(b)}));
    }
    static const Expr::Node &node(const Expr &e) { return *e.node_; }
    static bool is_const(const Expr &e, double v) {
        return e.node_->op == Op::constant && e.node_->value == v;
    }
    static bool is_const(const Expr &e) { return e.node_->op == Op::constant; }
};


namespace {

using M = ExprMaker;

const double kPi = 3.14159265358979323846;

/// A tensor of the shape and backend of `like` filled with `value`; without `like` (a constant
/// with no state to follow), a 1x1 one on the current backend.
CORA_FROM_NUMBERS Tensor aux_constant(const Tensor &like, double value) {
    if (!like.defined()) return Tensor::from_data({value}, {1, 1});
    return like.full_like(like.shape(), value);
}

/// The mask of `like`'s shape with 1 where the flag is set: a decision made on the host values,
/// applied as a product so that the values stay on the tensor.
Tensor aux_mask(const Tensor &like, const std::vector<double> &flags) {
    return Tensor::like(like, flags, like.shape());
}

/// a^n for a tensor and an integer n, by repeated squaring.
Tensor aux_power(const Tensor &a, int n) {
    if (n < 0) return aux_constant(a, 1.0).divide(aux_power(a, -n));
    if (n == 0) return aux_constant(a, 1.0);
    Tensor result, base = a;
    for (int k = n; k > 0; k >>= 1) {
        if (k & 1) result = result.defined() ? result.multiply(base) : base;
        if (k > 1) base = base.multiply(base);
    }
    return result;
}

// Ranges ------------------------------------------------------------------------------------------

/// Whether the interval [lo, hi] contains a point of the form offset + 2 pi k.
bool aux_has_peak(double lo, double hi, double offset) {
    return std::ceil((lo - offset) / (2 * kPi)) <= std::floor((hi - offset) / (2 * kPi));
}

/// The range of sin, or of cos (as cos x = sin(x + pi/2)), over [lo, hi]: the values at the
/// ends, moved to +-1 where a peak lies inside.
Range aux_trig(const Range &a, bool cosine) {
    const std::vector<double> lo = a.lo.data(), hi = a.hi.data();
    const double phase = cosine ? kPi / 2 : 0;
    std::vector<double> top(lo.size()), bottom(lo.size());
    std::vector<double> keep_top(lo.size()), keep_bottom(lo.size());
    for (std::size_t i = 0; i < lo.size(); ++i) {
        const double l = lo[i] + phase, h = hi[i] + phase;
        const bool wide = h - l >= 2 * kPi;
        top[i] = wide || aux_has_peak(l, h, kPi / 2);
        bottom[i] = wide || aux_has_peak(l, h, -kPi / 2);
        keep_top[i] = 1 - top[i];
        keep_bottom[i] = 1 - bottom[i];
    }
    const Tensor va = cosine ? a.lo.cos() : a.lo.sin(), vb = cosine ? a.hi.cos() : a.hi.sin();
    return {va.minimum(vb).multiply(aux_mask(a.lo, keep_bottom)) - aux_mask(a.lo, bottom),
            va.maximum(vb).multiply(aux_mask(a.lo, keep_top)) + aux_mask(a.lo, top)};
}

Range aux_div(const Range &a, const Range &b) {
    const std::vector<double> lo = b.lo.data(), hi = b.hi.data();
    // A divisor range that contains 0 has no enclosure.
    for (std::size_t i = 0; i < lo.size(); ++i)
        if (lo[i] <= 0 && hi[i] >= 0)
            CORA_FAIL(Error::out_of_domain,
                      "cora: an expression divides by a range that contains 0; "
                      "the box is too large for this dynamics");
    const Tensor one = aux_constant(b.lo, 1.0);
    return a * Range(one.divide(b.hi), one.divide(b.lo));
}

Range aux_pow(const Range &a, int n) {
    if (n < 0) return aux_div(Range(aux_constant(a.lo, 1.0)), aux_pow(a, -n));
    if (n == 0) return Range(aux_constant(a.lo, 1.0));
    const Tensor p = aux_power(a.lo, n), q = aux_power(a.hi, n);
    if (n % 2 == 1) return {p, q}; // odd powers grow
    // An even power dips to 0 where the range contains 0.
    const std::vector<double> lo = a.lo.data(), hi = a.hi.data();
    std::vector<double> keep(lo.size());
    for (std::size_t i = 0; i < lo.size(); ++i) keep[i] = !(lo[i] <= 0 && hi[i] >= 0);
    return {p.minimum(q).multiply(aux_mask(a.lo, keep)), p.maximum(q)};
}

/// The host bounds of a range, for a decision on the values.
std::pair<std::vector<double>, std::vector<double>> aux_bounds(const Range &a) {
    return {a.lo.data(), a.hi.data()};
}

/// a^p for a real exponent: increasing for p > 0 on a >= 0, decreasing for p < 0 on a > 0.
Range aux_rpow(const Range &a, double p) {
    for (const double l : a.lo.data())
        if (!(l > 0 || (p > 0 && l >= 0)))
            CORA_FAIL(Error::out_of_domain,
                      "cora: an expression takes a real power of a range that is not positive; "
                      "the box is too large for this dynamics");
    if (p == 0.5) return {a.lo.sqrt(), a.hi.sqrt()};
    if (p > 0) return {a.lo.pow(p), a.hi.pow(p)};
    return {a.hi.pow(p), a.lo.pow(p)};
}

/// tan is increasing between its poles pi/2 + k pi; a range that holds a pole has no enclosure.
Range aux_tan(const Range &a) {
    const auto [lo, hi] = aux_bounds(a);
    for (std::size_t i = 0; i < lo.size(); ++i)
        if (aux_has_peak(2 * lo[i], 2 * hi[i], kPi) || !(hi[i] - lo[i] < kPi))
            CORA_FAIL(Error::out_of_domain,
                      "cora: an expression takes tan of a range that holds a pole; the box is "
                      "too large for this dynamics");
    return {a.lo.tan(), a.hi.tan()};
}

// Evaluation --------------------------------------------------------------------------------------

/// The value of a node from the values of its operands.
Tensor aux_apply(const Expr::Node &n, const Tensor &a, const Tensor &b) {
    switch (n.op) {
    case Op::add:
        return a + b;
    case Op::sub:
        return a - b;
    case Op::mul:
        return a.multiply(b);
    case Op::div:
        return a.divide(b);
    // the unary operations
    case Op::neg:
        return a * -1.0;
    case Op::pow:
        return aux_power(a, n.index);
    case Op::rpow:
        return n.value == 0.5 ? a.sqrt() : a.pow(n.value);
    case Op::sin:
        return a.sin();
    case Op::cos:
        return a.cos();
    case Op::tan:
        return a.tan();
    case Op::exp:
        return a.exp();
    default:
        break;
    }
    CORA_FAIL(Error::report_to_dev, "cora: an expression node of an unknown operation");
}

/// The same over boxes.
Range aux_apply(const Expr::Node &n, const Range &a, const Range &b) {
    switch (n.op) {
    case Op::add:
        return a + b;
    case Op::sub:
        return a - b;
    case Op::mul:
        return a * b;
    case Op::div:
        return aux_div(a, b);
    // the unary operations
    case Op::neg:
        return {a.hi * -1.0, a.lo * -1.0};
    case Op::pow:
        return aux_pow(a, n.index);
    case Op::rpow:
        return aux_rpow(a, n.value);
    case Op::sin:
        return aux_trig(a, false);
    case Op::cos:
        return aux_trig(a, true);
    case Op::tan:
        return aux_tan(a);
    case Op::exp:
        return {a.lo.exp(), a.hi.exp()};
    default:
        break;
    }
    CORA_FAIL(Error::report_to_dev, "cora: an expression node of an unknown operation");
}

/// The value of an expression at the tensors or over the ranges x; `like` gives the constants
/// their shape. A product or quotient with a constant is a scaling, which saves the constant. A
/// node met again (derivatives share their operands) is looked up, not evaluated again.
template <class T>
struct Evaluator {
    const std::vector<T> &x;
    const Tensor &like;
    std::unordered_map<const Expr::Node *, T> memo = {};

    static T scaled(const T &a, double s) { return a * s; }

    T constant(double v) const {
        if constexpr (std::is_same_v<T, Tensor>)
            return aux_constant(like, v);
        else
            return Range(aux_constant(like, v));
    }

    // a constant is made, a node met before is looked up, any other one evaluated
    T operator()(const Expr &e) {
        const Expr::Node &n = M::node(e);
        if (n.op == Op::constant) return constant(n.value);
        if (n.op != Op::var) {
            const auto found = memo.find(&n);
            if (found != memo.end()) return found->second;
        }
        T value = node_value(n);
        if (n.op != Op::var) memo.emplace(&n, value);
        return value;
    }

    // a variable is its value; any other node is computed from its operands
    T node_value(const Expr::Node &n) {
        if (n.op == Op::var) {
            if (n.index >= static_cast<int>(x.size()))
                CORA_FAIL(Error::wrong_value,
                          "cora: the expression uses a variable the point does not have");
            return x[n.index];
        }
        // Products and quotients with a constant scale their other operand.
        const bool binary =
            n.op == Op::add || n.op == Op::sub || n.op == Op::mul || n.op == Op::div;
        if (n.op == Op::mul && M::is_const(n.b)) return scaled((*this)(n.a), M::node(n.b).value);
        if (n.op == Op::mul && M::is_const(n.a)) return scaled((*this)(n.b), M::node(n.a).value);
        if (n.op == Op::div && M::is_const(n.b) && M::node(n.b).value != 0)
            return scaled((*this)(n.a), 1 / M::node(n.b).value);
        const T a = (*this)(n.a);
        return aux_apply(n, a, binary ? (*this)(n.b) : a);
    }
};

/// The prototype of the constants: the first variable, or none.
template <class T>
Tensor aux_prototype(const std::vector<T> &x) {
    if (x.empty()) return Tensor();
    if constexpr (std::is_same_v<T, Tensor>)
        return x[0];
    else
        return x[0].lo;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Range ---------------------------------------------------------------------------------------

Range operator+(const Range &a, const Range &b) { return {a.lo + b.lo, a.hi + b.hi}; }

Range operator-(const Range &a, const Range &b) { return {a.lo - b.hi, a.hi - b.lo}; }

Range operator*(const Range &a, const Range &b) {
    const Tensor p[4] = {a.lo.multiply(b.lo), a.lo.multiply(b.hi), a.hi.multiply(b.lo),
                         a.hi.multiply(b.hi)};
    return {p[0].minimum(p[1]).minimum(p[2].minimum(p[3])),
            p[0].maximum(p[1]).maximum(p[2].maximum(p[3]))};
}

Range operator*(const Range &a, double s) {
    return s >= 0 ? Range(a.lo * s, a.hi * s) : Range(a.hi * s, a.lo * s);
}

Range square(const Range &a) {
    const Tensor p = a.lo.multiply(a.lo), q = a.hi.multiply(a.hi);
    // The smallest square is 0 if the range contains 0, else at the nearer end.
    const Tensor dist = a.lo.maximum(0.0) + (a.hi * -1.0).maximum(0.0);
    return {dist.multiply(dist), p.maximum(q)};
}

// Building ------------------------------------------------------------------------------------

Expr::Expr(double value)
    : node_(std::make_shared<const Node>(Node{Op::constant, value, 0, {}, {}})) {}

Expr Expr::var(int i) {
    if (i < 0) CORA_FAIL(Error::wrong_value, "cora: the index of a variable is 0, 1, 2, ...");
    return M::make(Op::var, Expr(), Expr(), 0, i);
}

// Every operation folds constants (0 + x, 1 * x, 2 * 3), which keeps derivatives small.
Expr operator+(const Expr &a, const Expr &b) {
    if (M::is_const(a, 0)) return b;
    if (M::is_const(b, 0)) return a;
    if (M::is_const(a) && M::is_const(b)) return Expr(M::node(a).value + M::node(b).value);
    return M::make(Op::add, a, b);
}

Expr operator-(const Expr &a, const Expr &b) {
    if (M::is_const(b, 0)) return a;
    if (M::is_const(a) && M::is_const(b)) return Expr(M::node(a).value - M::node(b).value);
    return M::make(Op::sub, a, b);
}

Expr operator*(const Expr &a, const Expr &b) {
    if (M::is_const(a, 0) || M::is_const(b, 0)) return Expr(0.0);
    if (M::is_const(a, 1)) return b;
    if (M::is_const(b, 1)) return a;
    if (M::is_const(a) && M::is_const(b)) return Expr(M::node(a).value * M::node(b).value);
    return M::make(Op::mul, a, b);
}

// Dividing by the constant 0 is an error here; a range around 0 is one in enclose.
Expr operator/(const Expr &a, const Expr &b) {
    if (M::is_const(b, 0))
        CORA_FAIL(Error::out_of_domain, "cora: an expression divides by the constant 0");
    if (M::is_const(a, 0)) return Expr(0.0);
    if (M::is_const(b, 1)) return a;
    return M::make(Op::div, a, b);
}

Expr operator-(const Expr &a) {
    if (M::is_const(a)) return Expr(-M::node(a).value);
    return M::make(Op::neg, a);
}

Expr pow(const Expr &a, int n) {
    if (n == 0) return Expr(1.0);
    if (n == 1) return a;
    return M::make(Op::pow, a, M::make(Op::constant), 0, n);
}

// An integer exponent is the integer power, whose range knows the sign of the base.
Expr pow(const Expr &a, double p) {
    if (p == std::round(p) && std::abs(p) < 1e9) return pow(a, static_cast<int>(p));
    if (M::is_const(a)) return Expr(std::pow(M::node(a).value, p));
    return M::make(Op::rpow, a, M::make(Op::constant), p);
}

Expr sqrt(const Expr &a) { return pow(a, 0.5); }

// The functions are plain nodes; diff knows their derivatives.
Expr sin(const Expr &a) { return M::make(Op::sin, a); }
Expr cos(const Expr &a) { return M::make(Op::cos, a); }
Expr tan(const Expr &a) { return M::make(Op::tan, a); }
Expr exp(const Expr &a) { return M::make(Op::exp, a); }

// Derivative and evaluation -------------------------------------------------------------------

Expr Expr::diff(int i) const {
    const Node &n = *node_;
    if (n.op == Op::constant) return Expr(0.0);
    if (n.op == Op::var) return Expr(n.index == i ? 1.0 : 0.0);
    const Expr da = n.a.diff(i); // the chain rule needs the inner derivative for every operation
    switch (n.op) {
    case Op::add:
        return da + n.b.diff(i);
    case Op::sub:
        return da - n.b.diff(i);
    case Op::mul:
        return da * n.b + n.a * n.b.diff(i);
    case Op::div:
        return (da * n.b - n.a * n.b.diff(i)) / (n.b * n.b);
    // the unary operations
    case Op::neg:
        return -da;
    case Op::pow:
        return Expr(double(n.index)) * pow(n.a, n.index - 1) * da;
    case Op::rpow:
        return Expr(n.value) * pow(n.a, n.value - 1) * da;
    case Op::sin:
        return cos(n.a) * da;
    case Op::cos:
        return -(sin(n.a) * da);
    case Op::tan:
        return (1 + pow(tan(n.a), 2)) * da; // CORA's symbolic derivative, tan^2 + 1
    case Op::exp:
        return exp(n.a) * da;
    default:
        break;
    }
    CORA_FAIL(Error::report_to_dev, "cora: an expression node of an unknown operation");
}

bool Expr::is_zero() const { return M::is_const(*this, 0); }

// The first variable gives the constants their shape and backend.
Tensor Expr::eval_tensor(const std::vector<Tensor> &x) const {
    const Tensor like = aux_prototype(x);
    return Evaluator<Tensor>{x, like}(*this);
}

CORA_FROM_NUMBERS double Expr::eval(const std::vector<double> &x) const {
    std::vector<Tensor> tensors;
    tensors.reserve(x.size());
    for (const double v : x) tensors.push_back(Tensor::from_data({v}, {1, 1}));
    return eval_tensor(tensors).data()[0];
}

Range Expr::enclose(const std::vector<Range> &x) const {
    const Tensor like = aux_prototype(x);
    return Evaluator<Range>{x, like}(*this);
}

// One evaluator for all, so that its table holds the subexpressions they share.
std::vector<Tensor> Expr::eval_tensor(const std::vector<Expr> &exprs,
                                      const std::vector<Tensor> &x) {
    const Tensor like = aux_prototype(x);
    Evaluator<Tensor> evaluate{x, like};
    std::vector<Tensor> out;
    out.reserve(exprs.size());
    for (const Expr &e : exprs) out.push_back(evaluate(e));
    return out;
}

std::vector<Range> Expr::enclose(const std::vector<Expr> &exprs, const std::vector<Range> &x) {
    const Tensor like = aux_prototype(x);
    Evaluator<Range> evaluate{x, like};
    std::vector<Range> out;
    out.reserve(exprs.size());
    for (const Expr &e : exprs) out.push_back(evaluate(e));
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
