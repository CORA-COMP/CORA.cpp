// test_level_set_split_equality - LevelSet split_equality: the equality and the inequality part
//
// CORA: contSet/@levelSet/splitEquality.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "level_set_testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        const Expr x = Expr::var(0);
        const LevelSet ls({x, x * x, x - 1.0}, 1, {CompOp::eq, CompOp::le, CompOp::eq});
        const auto [eq, ineq] = ls.split_equality();
        check(eq && eq->num_equations() == 2 && ineq && ineq->num_equations() == 1,
              b + ": two equalities, one inequality");
        check(ineq->comp_ops().front() == CompOp::le, b + ": the inequality keeps its operator");
        // only one kind: the other part is none
        const auto [e2, i2] = LevelSet(x, 1, CompOp::eq).split_equality();
        check(e2 && !i2, b + ": an equality set has no inequality part");
        const auto [e3, i3] = LevelSet(x, 1, CompOp::lt).split_equality();
        check(!e3 && i3, b + ": an inequality set has no equality part");
    });
    ChecksScope off(false);
    check(LevelSet(Expr::var(0), 1, CompOp::eq).split_equality().first.has_value(),
          "split_equality: checks off");
    return test::finish("level_set split_equality");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
