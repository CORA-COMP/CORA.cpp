// split_equality - the equality and the inequality part, as CORA's levelSet.splitEquality
//
// The first level set holds the '==' constraints, the second the others, each in its order.
// A part without constraints is none (CORA: []).
//
// Syntax:   auto [eq, ineq] = ls.split_equality();
// Inputs:   -
// Outputs:  eq, ineq - the two parts
// See also: complement
//
// CORA: contSet/@levelSet/splitEquality.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/level_set/level_set.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::pair<std::optional<LevelSet>, std::optional<LevelSet>> LevelSet::split_equality() const {
    detail::Entry entry;
    std::vector<Expr> eq_part, ineq_part;
    std::vector<CompOp> eq_ops, ineq_ops;
    // each constraint goes to the part of its comparison
    for (std::size_t i = 0; i < eq_.size(); ++i) {
        (ops_[i] == CompOp::eq ? eq_part : ineq_part).push_back(eq_[i]);
        (ops_[i] == CompOp::eq ? eq_ops : ineq_ops).push_back(ops_[i]);
    }
    std::pair<std::optional<LevelSet>, std::optional<LevelSet>> out;
    if (!eq_part.empty()) out.first.emplace(eq_part, dim(), eq_ops);
    if (!ineq_part.empty()) out.second.emplace(ineq_part, dim(), ineq_ops);
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
