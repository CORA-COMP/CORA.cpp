// expr - symbolic expressions of the state, the dynamics of a nonlinear system
//
// The trees of expressions, their derivative, and their evaluation at tensors and over boxes.
//
// Syntax:   see expr.h
// See also: contDynamics/nonlinearSys/nonlinearSys.h

#include "global/expr.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <type_traits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

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

/// A tensor of the shape and backend of `like` filled with `value`; without `like`, a 1x1 one.
Tensor aux_constant(const Tensor &like, double value) {
    if (!like.defined()) return Tensor::fromData({value}, {1, 1});
    int64_t count = 1;
    for (const int64_t d : like.shape()) count *= d;
    return Tensor::like(like, std::vector<double>(count, value), like.shape());
}

/// The mask of `like`'s shape with 1 where the flag is set: a decision made on the host values,
/// applied as a product so that the values stay on the tensor.
Tensor aux_mask(const Tensor &like, const std::vector<double> &flags) {
    return Tensor::like(like, flags, like.shape());
}

/// a^n for a tensor and an integer n, by repeated squaring.
Tensor aux_power(const Tensor &a, int n) {
    if (n < 0) return aux_constant(a, 1.0).div(aux_power(a, -n));
    if (n == 0) return aux_constant(a, 1.0);
    Tensor result, base = a;
    for (int k = n; k > 0; k >>= 1) {
        if (k & 1) result = result.defined() ? result.mul(base) : base;
        if (k > 1) base = base.mul(base);
    }
    return result;
}

// Ranges ------------------------------------------------------------------------------------------

/// Whether the interval [lo, hi] contains a point of the form offset + 2 pi k.
bool aux_hasPeak(double lo, double hi, double offset) {
    return std::ceil((lo - offset) / (2 * kPi)) <= std::floor((hi - offset) / (2 * kPi));
}

/// The range of sin, or of cos (as cos x = sin(x + pi/2)), over [lo, hi]: the values at the
/// ends, moved to +-1 where a peak lies inside.
Range aux_trig(const Range &a, bool cosine) {
    const std::vector<double> lo = a.lo.data(), hi = a.hi.data();
    const double phase = cosine ? kPi / 2 : 0;
    std::vector<double> top(lo.size()), bottom(lo.size());
    std::vector<double> keepTop(lo.size()), keepBottom(lo.size());
    for (std::size_t i = 0; i < lo.size(); ++i) {
        const double l = lo[i] + phase, h = hi[i] + phase;
        const bool wide = h - l >= 2 * kPi;
        top[i] = wide || aux_hasPeak(l, h, kPi / 2);
        bottom[i] = wide || aux_hasPeak(l, h, -kPi / 2);
        keepTop[i] = 1 - top[i];
        keepBottom[i] = 1 - bottom[i];
    }
    const Tensor va = cosine ? a.lo.cos() : a.lo.sin(), vb = cosine ? a.hi.cos() : a.hi.sin();
    return {Tensor::minimum(va, vb).mul(aux_mask(a.lo, keepBottom)) - aux_mask(a.lo, bottom),
            Tensor::maximum(va, vb).mul(aux_mask(a.lo, keepTop)) + aux_mask(a.lo, top)};
}

Range aux_div(const Range &a, const Range &b) {
    const std::vector<double> lo = b.lo.data(), hi = b.hi.data();
    // A divisor range that contains 0 has no enclosure.
    for (std::size_t i = 0; i < lo.size(); ++i)
        if (lo[i] <= 0 && hi[i] >= 0)
            throw std::domain_error("cora: an expression divides by a range that contains 0; "
                                    "the box is too large for this dynamics");
    const Tensor one = aux_constant(b.lo, 1.0);
    return a * Range(one.div(b.hi), one.div(b.lo));
}

Range aux_pow(const Range &a, int n) {
    if (n < 0) return aux_div(Range(aux_constant(a.lo, 1.0)), aux_pow(a, -n));
    if (n == 0) return Range(aux_constant(a.lo, 1.0));
    const Tensor p = aux_power(a.lo, n), q = aux_power(a.hi, n);
    if (n % 2 == 1) return {p, q};  // odd powers grow
    // An even power dips to 0 where the range contains 0.
    const std::vector<double> lo = a.lo.data(), hi = a.hi.data();
    std::vector<double> keep(lo.size());
    for (std::size_t i = 0; i < lo.size(); ++i) keep[i] = !(lo[i] <= 0 && hi[i] >= 0);
    return {Tensor::minimum(p, q).mul(aux_mask(a.lo, keep)), Tensor::maximum(p, q)};
}

// Evaluation --------------------------------------------------------------------------------------

/// The value of a node from the values of its operands.
Tensor aux_apply(const Expr::Node &n, const Tensor &a, const Tensor &b) {
    switch (n.op) {
    case Op::Add: return a + b;
    case Op::Sub: return a - b;
    case Op::Mul: return a.mul(b);
    case Op::Div: return a.div(b);
    case Op::Neg: return a * -1.0;
    case Op::Pow: return aux_power(a, n.index);
    case Op::Sin: return a.sin();
    case Op::Cos: return a.cos();
    case Op::Exp: return a.exp();
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
    case Op::Neg: return {a.hi * -1.0, a.lo * -1.0};
    case Op::Pow: return aux_pow(a, n.index);
    case Op::Sin: return aux_trig(a, false);
    case Op::Cos: return aux_trig(a, true);
    case Op::Exp: return {a.lo.exp(), a.hi.exp()};
    default: break;
    }
    throw std::logic_error("cora: an expression node of an unknown operation");
}

/// The value of an expression at the tensors or over the ranges x; `like` gives the constants
/// their shape. A product or quotient with a constant is a scaling, which saves the constant.
template <class T> struct Evaluator {
    const std::vector<T> &x;
    const Tensor &like;

    static T scaled(const T &a, double s) { return a * s; }

    T constant(double v) const {
        if constexpr (std::is_same_v<T, Tensor>) return aux_constant(like, v);
        else return Range(aux_constant(like, v));
    }

    T operator()(const Expr &e) const {
        const Expr::Node &n = M::node(e);
        if (n.op == Op::Const) return constant(n.value);
        if (n.op == Op::Var) {
            if (n.index >= static_cast<int>(x.size()))
                throw std::out_of_range(
                    "cora: the expression uses a variable the point does not have");
            return x[n.index];
        }
        // Products and quotients with a constant scale their other operand.
        const bool binary =
            n.op == Op::Add || n.op == Op::Sub || n.op == Op::Mul || n.op == Op::Div;
        if (n.op == Op::Mul && M::isConst(n.b))
            return scaled((*this)(n.a), M::node(n.b).value);
        if (n.op == Op::Mul && M::isConst(n.a))
            return scaled((*this)(n.b), M::node(n.a).value);
        if (n.op == Op::Div && M::isConst(n.b) && M::node(n.b).value != 0)
            return scaled((*this)(n.a), 1 / M::node(n.b).value);
        const T a = (*this)(n.a);
        return aux_apply(n, a, binary ? (*this)(n.b) : a);
    }
};

/// The prototype of the constants: the first variable, or none.
template <class T> Tensor aux_prototype(const std::vector<T> &x) {
    if (x.empty()) return Tensor();
    if constexpr (std::is_same_v<T, Tensor>) return x[0];
    else return x[0].lo;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Range ---------------------------------------------------------------------------------------

Range operator+(const Range &a, const Range &b) { return {a.lo + b.lo, a.hi + b.hi}; }

Range operator-(const Range &a, const Range &b) { return {a.lo - b.hi, a.hi - b.lo}; }

Range operator*(const Range &a, const Range &b) {
    const Tensor p[4] = {a.lo.mul(b.lo), a.lo.mul(b.hi), a.hi.mul(b.lo), a.hi.mul(b.hi)};
    return {Tensor::minimum(Tensor::minimum(p[0], p[1]), Tensor::minimum(p[2], p[3])),
            Tensor::maximum(Tensor::maximum(p[0], p[1]), Tensor::maximum(p[2], p[3]))};
}

Range operator*(const Range &a, double s) {
    return s >= 0 ? Range(a.lo * s, a.hi * s) : Range(a.hi * s, a.lo * s);
}

Range square(const Range &a) {
    const Tensor p = a.lo.mul(a.lo), q = a.hi.mul(a.hi);
    // The smallest square is 0 where the range contains 0, else the smaller end: with
    // dist = max(lo, 0) + max(-hi, 0) the distance of 0 from the range.
    const Tensor dist = a.lo.pos() + (a.hi * -1.0).pos();
    return {dist.mul(dist), Tensor::maximum(p, q)};
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

bool Expr::isZero() const { return M::isConst(*this, 0); }

// The first variable gives the constants their shape and backend.
Tensor Expr::evalTensor(const std::vector<Tensor> &x) const {
    const Tensor like = aux_prototype(x);
    return Evaluator<Tensor>{x, like}(*this);
}

double Expr::eval(const std::vector<double> &x) const {
    std::vector<Tensor> tensors;
    for (const double v : x) tensors.push_back(Tensor::fromData({v}, {1, 1}));
    return evalTensor(tensors).data()[0];
}

Range Expr::enclose(const std::vector<Range> &x) const {
    const Tensor like = aux_prototype(x);
    return Evaluator<Range>{x, like}(*this);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
