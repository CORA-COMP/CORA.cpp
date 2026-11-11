// test_level_set_is_intersecting - LevelSet is_intersecting: boxes, with the '==' rows decided
//
// CORA: contSet/@levelSet/isIntersecting_.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "level_set_testing.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        const LevelSet circ(test::circle(0, 0, 1), 2, CompOp::eq);
        // the box [1, 2] x [0, 0] touches the circle at (1, 0): inf f = 0 (CORA missed this)
        check(circ.is_intersecting(Interval(column({1.0, 0.0}), column({2.0, 0.0}))),
              b + ": a box that touches the circle");
        check(!circ.is_intersecting(test::box({2, 2}, {3, 3})), b + ": a box outside");
        check(!circ.is_intersecting(test::box({-0.1, -0.1}, {0.1, 0.1})), b + ": a box inside");
        check(circ.is_intersecting(test::box({0, 0}, {2, 2})), b + ": a box that crosses");
        // '<=' and '<' differ at inf f = 0
        const LevelSet disc(test::circle(0, 0, 1), 2, CompOp::le),
            open(test::circle(0, 0, 1), 2, CompOp::lt);
        const Interval touch(column({1.0, 0.0}), column({2.0, 0.0}));
        check(disc.is_intersecting(touch) && !open.is_intersecting(touch), b + ": '<=' versus '<'");
        // any set goes through its box
        const Zonotope Z(column({3.0, 3.0}), Tensor::from_data({0.5, 0, 0, 0.5}, {2, 2}));
        check(!circ.is_intersecting(Z), b + ": a zonotope outside");
        // all rows must pass: the circle and the half-plane x <= -2 do not meet the box
        const LevelSet two({test::circle(0, 0, 1), Expr::var(0) + 2.0}, 2,
                           {CompOp::eq, CompOp::le});
        check(!two.is_intersecting(test::box({0, 0}, {2, 2})), b + ": two rows");
    });
    const LevelSet circ(test::circle(0, 0, 1), 2, CompOp::eq);
    check(test::raises([&] { circ.is_intersecting(Interval(column({0.0}), column({1.0}))); },
                       Error::dimension_mismatch),
          "is_intersecting: dimension");
    ChecksScope off(false);
    check(circ.is_intersecting(test::box({0, 0}, {2, 2})),
          "is_intersecting: valid with checks off");
    return test::finish("level_set is_intersecting");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
