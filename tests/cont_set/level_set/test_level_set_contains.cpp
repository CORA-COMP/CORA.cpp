// test_level_set_contains - LevelSet contains: points and boxes, with the '==' rows decided
//
// CORA: contSet/@levelSet/contains_.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
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
        // the unit disc f <= 0
        const LevelSet disc(test::circle(0, 0, 1), 2, CompOp::le);
        check(disc.contains(Tensor::from_data({0, 1, 0.5, 0}, {2, 2})), b + ": points of the disc");
        check(!disc.contains(Tensor::from_data({2, 0}, {2, 1})), b + ": a point outside");
        check(disc.contains(test::box({-0.5, -0.5}, {0.5, 0.5})), b + ": a box inside");
        check(!disc.contains(test::box({0, 0}, {1, 1})), b + ": a box that leaves");
        // the open disc excludes the circle itself
        const LevelSet open(test::circle(0, 0, 1), 2, CompOp::lt);
        check(!open.contains(test::box({0, 0}, {0.8, 0.8})) &&
                  open.contains(test::box({0, 0}, {0.5, 0.5})),
              b + ": '<' needs sup f < 0");
        // a circle holds its points, and a box only if f vanishes on it (CORA accepted sup f < 0)
        const LevelSet circ(test::circle(0, 0, 1), 2, CompOp::eq);
        check(circ.contains(Tensor::from_data({1, 0}, {2, 1})), b + ": a point on the circle");
        check(!circ.contains(test::box({0, 0}, {0.1, 0.1})), b + ": '==' rejects a box with f < 0");
        check(circ.contains(test::box({1, 0}, {1, 0})), b + ": '==' accepts a point box on it");
        // the empty box is inside every level set
        check(disc.contains(Interval::empty(2)), b + ": the empty box");
    });
    const LevelSet disc(test::circle(0, 0, 1), 2, CompOp::le);
    check(test::raises([&] { disc.contains(Tensor::zeros({3, 1})); }, Error::dimension_mismatch),
          "contains: points of another dimension");
    check(test::raises([&] { disc.contains(test::box({0}, {1})); }, Error::dimension_mismatch),
          "contains: a box of another dimension");
    ChecksScope off(false);
    check(disc.contains(Tensor::zeros({2, 1})), "contains: valid with checks off");
    return test::finish("level_set contains");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
