// test_level_set_lift - LevelSet lift: the set in more dimensions
//
// CORA: contSet/@levelSet/lift_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   24-November-2026
// Reviewed:      ---

#include "level_set_testing.h"

using namespace cora;
using test::check;
using test::close;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        // the unit circle of the states (0, 1) on the states (2, 0) of three: x2^2 + x0^2 = 1
        const LevelSet ls(test::circle(0, 0, 1), 2, CompOp::eq);
        const Set lifted = ls.lift(3, std::vector<int64_t>{2, 0});
        check(lifted.is<LevelSet>() && lifted->dim() == 3, b + ": three states");
        const LevelSet &L = lifted.as<LevelSet>();
        // the point (x0, x1, x2) = (0.6, 7, 0.8) is on it for any x1
        check(close(L.fun_han(Tensor::from_data({0.6, 7.0, 0.8}, {1, 3})), {0.0}, 1e-12),
              b + ": the new state is free");
        check(close(L.fun_han(Tensor::from_data({1.0, -3.0, 0.0}, {1, 3})), {0.0}, 1e-12),
              b + ": a second point");
        // every elementary function renames its argument: f(x0) on the state 2 of three
        const Expr x = Expr::var(0);
        const std::vector<Expr> fs = {log(x),  asin(x), acos(x),  atan(x),  sinh(x),
                                      cosh(x), tanh(x), asinh(x), acosh(x), atanh(x)};
        for (const Expr &f : fs) {
            const Set moved = LevelSet(f, 1, CompOp::eq).lift(3, std::vector<int64_t>{2});
            const double arg = (&f == &fs[8]) ? 1.5 : 0.4; // acosh needs at least 1
            check(close(moved.as<LevelSet>().fun_han(Tensor::from_data({9.0, 9.0, arg}, {1, 3})),
                        {f.eval({arg})}, 1e-12),
                  b + ": an elementary function is renamed");
        }
    });
    const LevelSet ls(test::circle(0, 0, 1), 2, CompOp::eq);
    check(test::raises([&] { ls.lift(1, std::vector<int64_t>{0, 1}); }, Error::wrong_value),
          "lift: fewer states");
    check(test::raises([&] { ls.lift(3, std::vector<int64_t>{0, 5}); }, Error::wrong_value),
          "lift: a position outside");
    ChecksScope off(false);
    check(ls.lift(3, std::vector<int64_t>{0, 1})->dim() == 3, "lift: checks off");
    return test::finish("level_set lift");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
