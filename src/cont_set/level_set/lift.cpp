// lift - a level set in more dimensions, as CORA's levelSet.lift_
//
// State j of the set becomes state dims[j] of the N states; the other states do not occur in the
// constraints, so they are free. The expressions are rebuilt with the renamed variables.
//
// Syntax:   Set L = ls.lift(N, dims);
// Inputs:   N - the new dimension
//           dims - the new position of each state (0-based)
// Outputs:  L - the level set of N states
// See also: complement
//
// CORA: contSet/@levelSet/lift_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/level_set/level_set.h"
#include "global/format.h"

#include <unordered_map>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using Memo = std::unordered_map<const void *, Expr>;

/// The expression with state j renamed to state dims[j].
Expr aux_rename(const Expr &e, const std::vector<int64_t> &dims, Memo &memo) {
    if (const auto it = memo.find(e.id()); it != memo.end()) return it->second;
    const ExprView v = e.view();
    // a leaf is rebuilt, an operation from its renamed operands
    Expr out(0.0);
    if (v.op == ExprOp::constant) {
        out = Expr(v.value);
    } else if (v.op == ExprOp::var) {
        out = Expr::var(int(dims[std::size_t(v.index)]));
    } else {
        const Expr a = aux_rename(v.a, dims, memo);
        // the second operand exists only for the binary operations
        switch (v.op) {
        case ExprOp::add:
            out = a + aux_rename(v.b, dims, memo);
            break;
        case ExprOp::sub:
            out = a - aux_rename(v.b, dims, memo);
            break;
        // products and quotients rename both operands too
        case ExprOp::mul:
            out = a * aux_rename(v.b, dims, memo);
            break;
        case ExprOp::div:
            out = a / aux_rename(v.b, dims, memo);
            break;
        // the unary operations keep their exponent or argument
        case ExprOp::neg:
            out = -a;
            break;
        case ExprOp::pow:
            out = pow(a, v.index);
            break;
        case ExprOp::rpow:
            out = pow(a, v.value);
            break;
        // the transcendental functions rename their argument
        case ExprOp::sin:
            out = sin(a);
            break;
        case ExprOp::cos:
            out = cos(a);
            break;
        case ExprOp::tan:
            out = tan(a);
            break;
        case ExprOp::exp:
            out = exp(a);
            break;
        default:
            CORA_FAIL(Error::report_to_dev, "lift: an expression node of an unknown operation");
        }
    }
    memo.emplace(e.id(), out);
    return out;
}

} // namespace


// ===========================================  MAIN  =========================================== //

Set LevelSet::lift_impl(int64_t N, const std::vector<int64_t> &proj) const {
    Memo memo;
    std::vector<Expr> eq;
    eq.reserve(eq_.size());
    for (const Expr &f : eq_) eq.push_back(aux_rename(f, proj, memo));
    return LevelSet(std::move(eq), N, ops_);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
