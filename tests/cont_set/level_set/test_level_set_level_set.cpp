// test_level_set_level_set - LevelSet: the constructor, the dimension, the constraints
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

#include "level_set_testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        // the unit circle: f(0, 0) = -1, f(1, 0) = 0, f(2, 0) = 3
        const LevelSet ls(test::circle(0, 0, 1), 2, CompOp::eq);
        check(ls.dim() == 2 && ls.num_equations() == 1, b + ": dimension and constraints");
        const Tensor pts = Tensor::from_data({0, 1, 2, 0, 0, 0}, {2, 3});
        check(close(ls.fun_han(pts), {-1.0, 0.0, 3.0}), b + ": fun_han at points");
        // the gradient (2x, 2y) of the first constraint
        check(close(ls.gradient(0, 0).eval({3.0, 4.0}), 6.0) &&
                  close(ls.gradient(0, 1).eval({3.0, 4.0}), 8.0),
              b + ": the partial derivatives");
        // the box [0, 1] x [0, 1] has f in [-1, 1]
        const Interval F = ls.fun_han(test::box({0, 0}, {1, 1}));
        check(close(F.inf(), {-1.0}) && close(F.sup(), {1.0}), b + ": fun_han over a box");
        // empty and Inf: 1 <= 0 and -1 <= 0
        check(close(LevelSet::empty(2).fun_han(pts), {1.0, 1.0, 1.0}), b + ": the empty set");
        check(close(LevelSet::inf(2).fun_han(pts), {-1.0, -1.0, -1.0}), b + ": the whole space");
    });
    check(test::raises([] { LevelSet({}, 2, std::vector<CompOp>{}); },
                       Error::wrong_input_in_constructor),
          "LevelSet: no constraint");
    check(test::raises([] { LevelSet({Expr::var(0)}, 1, {CompOp::le, CompOp::eq}); },
                       Error::wrong_input_in_constructor),
          "LevelSet: a comparison per constraint");
    check(test::raises([] { LevelSet(Expr::var(2), 2, CompOp::le); },
                       Error::wrong_input_in_constructor),
          "LevelSet: a state beyond the dimension");
    const LevelSet ls(test::circle(0, 0, 1), 2, CompOp::eq);
    check(test::raises([&] { ls.fun_han(Tensor::zeros({3, 1})); }, Error::dimension_mismatch),
          "fun_han: rows");
    check(test::raises([&] { ls.gradient(1, 0); }, Error::wrong_value), "gradient: index");
    ChecksScope off(false);
    check(ls.fun_han(Tensor::zeros({2, 1})).data()[0] == -1.0, "fun_han: valid with checks off");
    return test::finish("level_set level_set");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
