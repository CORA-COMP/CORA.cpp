// level_set - the constructors of a level set, checked, as CORA's levelSet(eq, vars, compOp)
//
// The constraints are expressions of the states 0..n-1; each state a constraint names must exist.
// The partial derivatives are formed here, once (CORA forms them in the constructor as well).
// fun_han evaluates the constraints at points (CORA's funHan) or over a box.
//
// Syntax:   LevelSet ls(eq, n, ops);   LevelSet ls(f, n, op);   ls.fun_han(points);
// Inputs:   eq - the constraints f_i
//           n - the number of states
//           ops - one comparison per f_i
// Outputs:  ls - the level set
// See also: contains, is_intersecting
//
// CORA: contSet/@levelSet/levelSet.m
// CORA: contSet/@levelSet/dim.m
// CORA: contSet/@levelSet/empty.m
// CORA: contSet/@levelSet/Inf.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/level_set/level_set.h"
#include "global/format.h"

#include <algorithm>
#include <unordered_set>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The largest state index that the expression names (-1 for none); shared subtrees count once.
int aux_max_var(const Expr &e, std::unordered_set<const void *> &seen) {
    if (!seen.insert(e.id()).second) return -1;
    const ExprView v = e.view();
    if (v.op == ExprOp::var) return v.index;
    if (v.op == ExprOp::constant) return -1;
    int m = aux_max_var(v.a, seen);
    // only the binary operations have a second operand
    if (v.op == ExprOp::add || v.op == ExprOp::sub || v.op == ExprOp::mul || v.op == ExprOp::div)
        m = std::max(m, aux_max_var(v.b, seen));
    return m;
}

/// The first reason the constraints are refused, or "".
std::string aux_refusal(const std::vector<Expr> &eq, int64_t n, std::size_t ops) {
    if (eq.empty()) return "at least one constraint is needed";
    if (eq.size() != ops)
        return "there are " + std::to_string(eq.size()) + " constraints but " +
               std::to_string(ops) + " comparisons";
    for (const Expr &f : eq) {
        std::unordered_set<const void *> seen;
        if (aux_max_var(f, seen) >= n)
            return "a constraint names a state beyond the " + std::to_string(n) + " states";
    }
    return "";
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The constructors --------------------------------------------------------------------------------

LevelSet::LevelSet(std::vector<Expr> eq, int64_t n, std::vector<CompOp> ops)
    : DimensionSet(kClassId, n), eq_(std::move(eq)), ops_(std::move(ops)) {
    detail::Entry entry;
    const std::string why = aux_refusal(eq_, n, ops_.size());
    CORA_CHECK(why.empty(), Error::wrong_input_in_constructor, [&] { return "LevelSet: " + why; });
    // d f_i / d x_j for every pair
    for (const Expr &f : eq_) {
        std::vector<Expr> row;
        row.reserve(std::size_t(n));
        for (int64_t j = 0; j < n; ++j) row.push_back(f.diff(int(j)));
        grad_.push_back(std::move(row));
    }
}

LevelSet::LevelSet(const Expr &eq, int64_t n, CompOp op)
    : LevelSet(std::vector<Expr>{eq}, n, {op}) {}

LevelSet LevelSet::empty(int64_t n) { return {Expr(1.0), n, CompOp::le}; }

LevelSet LevelSet::inf(int64_t n) { return {Expr(-1.0), n, CompOp::le}; }

// The constraints ---------------------------------------------------------------------------------

const Expr &LevelSet::gradient(int64_t i, int64_t j) const {
    detail::Entry entry;
    CORA_CHECK(i >= 0 && i < num_equations() && j >= 0 && j < dim(), Error::wrong_value, [&] {
        return "gradient: the constraint " + std::to_string(i) + " and state " + std::to_string(j) +
               " must lie in " + std::to_string(num_equations()) + " x " + std::to_string(dim());
    });
    return grad_[std::size_t(i)][std::size_t(j)];
}

Tensor LevelSet::fun_han(const Tensor &x) const {
    detail::Entry entry;
    CORA_CHECK(x.ndim() >= 2 && x.shape()[std::size_t(x.ndim() - 2)] == dim(),
               Error::dimension_mismatch,
               [&] { return "fun_han: the points must have " + std::to_string(dim()) + " rows"; });
    // one tensor per state, all of one shape, then one row per constraint
    const int rows = int(x.ndim()) - 2;
    std::vector<Tensor> state;
    state.reserve(std::size_t(dim()));
    for (int64_t j = 0; j < dim(); ++j) state.push_back(x.slice(rows, j, j + 1));
    return Tensor::concat(Expr::eval_tensor(eq_, state), rows);
}

Interval LevelSet::fun_han(const Interval &I) const {
    detail::Entry entry;
    // the bounds of the box as the ranges of the states
    CORA_CHECK(I.inf().ndim() == 2 && I.inf().shape() == std::vector<int64_t>({dim(), 1}),
               Error::dimension_mismatch, [&] {
                   return "fun_han: the box must be a column of " + std::to_string(dim()) +
                          " states";
               });
    std::vector<Range> state;
    state.reserve(std::size_t(dim()));
    for (int64_t j = 0; j < dim(); ++j)
        state.emplace_back(I.inf().slice(0, j, j + 1), I.sup().slice(0, j, j + 1));
    std::vector<Tensor> lo, hi;
    lo.reserve(eq_.size());
    hi.reserve(eq_.size());
    for (const Range &r : Expr::enclose(eq_, state)) {
        lo.push_back(r.lo);
        hi.push_back(r.hi);
    }
    return {Tensor::concat(lo, 0), Tensor::concat(hi, 0)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
