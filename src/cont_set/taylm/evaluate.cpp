// evaluate - an expression on Taylor models, CORA applying a function handle to a taylm
//
// Sums, differences, products and integer powers are the operations of the models, so a
// polynomial keeps its dependency on the variables. A division by a constant is a product. Any
// other operation (sin, cos, tan, exp, a real power, a division by a model) is evaluated on the
// range of its operand: the result is a constant model with that range as remainder, an
// enclosure that has forgotten the dependency (CORA expands them in Taylor series). A subtree
// that occurs more than once is evaluated once.
//
// Syntax:   T = Taylm::evaluate(f, x);
// Inputs:   f - the expression
//           x - one model per variable of f, of the same variables
// Outputs:  T - the model that encloses the range of f
// See also: interval, Expr::view
//
// CORA.cpp only: CORA applies the function handle of the dynamics to the taylm objects directly
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/taylm/taylm.h"
#include "global/format.h"

#include <unordered_map>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The walk over the tree --------------------------------------------------------------------------

using Memo = std::unordered_map<const void *, Taylm>;

/// The range-based fallback: the operation applied to the ranges of its operands, as a model.
CORA_FROM_NUMBERS Taylm aux_range_op(const ExprView &v, const Taylm &a, const Taylm *b) {
    // the operation on fresh variables 0 and 1, enclosed over the operand ranges
    Expr x = Expr::var(0), y = Expr::var(1), e = x;
    switch (v.op) {
    case ExprOp::div:
        e = x / y;
        break;
    // the unary operations are the same function of the first operand
    case ExprOp::rpow:
        e = pow(x, v.value);
        break;
    case ExprOp::sin:
        e = sin(x);
        break;
    // sin, cos, tan and exp are enclosed over the range of the operand
    case ExprOp::cos:
        e = cos(x);
        break;
    case ExprOp::tan:
        e = tan(x);
        break;
    // a power with an integer exponent is the same function of the operand
    case ExprOp::exp:
        e = exp(x);
        break;
    case ExprOp::pow:
        e = pow(x, v.index);
        break;
    default:
        CORA_FAIL(Error::report_to_dev, "Taylm::evaluate: an operation without a range rule");
    }
    // the ranges of the operands, a unary operation repeats the first
    const auto ra = a.bounds();
    const auto rb = b ? b->bounds() : ra;
    std::vector<Range> box = {
        Range(Tensor::from_data({ra[0]}, {1, 1}), Tensor::from_data({ra[1]}, {1, 1})),
        Range(Tensor::from_data({rb[0]}, {1, 1}), Tensor::from_data({rb[1]}, {1, 1}))};
    const Range r = e.enclose(box);
    return Taylm::from_range(r.lo.data()[0], r.hi.data()[0], a.num_vars(), a.max_order());
}

/// The model of the subtree e at the models x of the variables.
Taylm aux_eval(const Expr &e, const std::vector<Taylm> &x, Memo &memo) {
    if (const auto it = memo.find(e.id()); it != memo.end()) return it->second;
    const ExprView v = e.view();
    const int64_t vars = x.front().num_vars();
    const int order = x.front().max_order();
    Taylm out = Taylm::constant(0.0, vars, order);
    // the leaves and the operations of models; the rest uses the range of its operand
    switch (v.op) {
    case ExprOp::constant:
        out = Taylm::constant(v.value, vars, order);
        break;
    case ExprOp::var:
        out = x[std::size_t(v.index)];
        break;
    // sums, differences and products are the operations of the models
    case ExprOp::add:
        out = aux_eval(v.a, x, memo).plus(aux_eval(v.b, x, memo));
        break;
    case ExprOp::sub:
        out = aux_eval(v.a, x, memo).minus(aux_eval(v.b, x, memo));
        break;
    case ExprOp::mul:
        out = aux_eval(v.a, x, memo).times(aux_eval(v.b, x, memo));
        break;
    case ExprOp::neg:
        out = aux_eval(v.a, x, memo).uminus();
        break;
    // a power by repeated products, a negative one by the range of its base
    case ExprOp::pow: {
        // a negative power is a division: only the range of the base is used
        const Taylm base = aux_eval(v.a, x, memo);
        out = v.index >= 0 ? base.power(v.index) : aux_range_op(v, base, nullptr);
        break;
    }
    case ExprOp::div: {
        const Taylm den = aux_eval(v.b, x, memo);
        const auto r = den.bounds();
        // a divisor that is a nonzero constant is a product with its inverse
        if (den.coefficients().size() <= 1 && r[0] == r[1] && r[0] != 0.0)
            out = aux_eval(v.a, x, memo).times(1.0 / r[0]);
        else
            out = aux_range_op(v, aux_eval(v.a, x, memo), &den);
        break;
    }
    default:
        out = aux_range_op(v, aux_eval(v.a, x, memo), nullptr);
        break;
    }
    memo.emplace(e.id(), out);
    return out;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The entry ---------------------------------------------------------------------------------------

Taylm Taylm::evaluate(const Expr &f, const std::vector<Taylm> &x) {
    detail::Entry entry;
    CORA_CHECK(!x.empty(), Error::wrong_value,
               [] { return std::string("Taylm::evaluate: at least one model is needed"); });
    for (const Taylm &t : x)
        CORA_CHECK(t.num_vars() == x.front().num_vars(), Error::dimension_mismatch, [] {
            return std::string("Taylm::evaluate: the models must range over the same variables");
        });
    Memo memo;
    return aux_eval(f, x, memo);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
