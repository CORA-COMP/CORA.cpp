// test_ellipsoid_mink_diff - Ellipsoid mink_diff: vector, points, nesting, refused general case
//
// CORA: contSet/@ellipsoid/minkDiff.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "ellipsoid_testing.h"

#include "cora/cont_set/fullspace/fullspace.h"
#include <cmath>
#include <limits>
#include <numbers>

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed: the shift, two points, a point minuend, a point S2, a smaller minuend, equal
/// shapes.
void mink_diff_known(const std::string &b) {
    const Ellipsoid E(Tensor({{4.0, 0.0}, {0.0, 1.0}}), column({1.0, 2.0}));
    check(close(E.mink_diff(column({1.0, 1.0})).q(), {0, 1}, 0), b + ": a vector");
    const Ellipsoid point(Tensor::zeros({2, 2}), column({1.0, 1.0})),
        other(Tensor::zeros({2, 2}), column({0.0, 1.0}));
    check(close(point.mink_diff(other).q(), {1, 0}, 0) &&
              point.mink_diff(other).Q().abs().max().item() == 0,
          b + ": two points");
    check(point.mink_diff(E).Q().numel() == 0, b + ": a point minuend of a proper S2 is empty");
    check(close(E.mink_diff(point).q(), {0, 1}, 0) && close(E.mink_diff(point).Q(), E.Q(), 0),
          b + ": a point S2");
    check(E.mink_diff(Ellipsoid(Tensor::eye(2) * 9.0)).Q().numel() == 0,
          b + ": a larger subtrahend is empty");
    check(E.mink_diff(E).Q().abs().max().item() == 0 && close(E.mink_diff(E).q(), {0, 0}, 0),
          b + ": E - E is the point 0");
    check(E.minkowski_difference(E).is<Ellipsoid>() &&
              E.minkowski_difference(Ellipsoid::empty(2)).is<Fullspace>(),
          b + ": the generic entry");
}

/// A smaller proper subtrahend needs directions of the partition; other types and classes are
/// refused.
void checks_and_their_switch() {
    const Ellipsoid E(Tensor({{4.0, 0.0}, {0.0, 1.0}}), column({1.0, 2.0}));
    const Ellipsoid small(Tensor::eye(2) * 0.25, column({0.0, 0.0}));
    check(test::raises([&] { E.mink_diff(small); }, Error::not_supported),
          "mink_diff: the general case");
    check(test::raises([&] { E.mink_diff(column({1.0})); }, Error::dimension_mismatch),
          "mink_diff: vector");
    check(test::raises([&] { E.mink_diff(Ellipsoid(Tensor::eye(3))); }, Error::dimension_mismatch),
          "mink_diff: dimension");
    check(test::raises([&] { E.minkowski_difference(small, "best"); }, Error::wrong_value),
          "mink_diff: type");
    check(test::raises([&] { E.minkowski_difference(Fullspace(2)); }, Error::noops),
          "mink_diff: class");
    ChecksScope off(false);
    check(E.mink_diff(column({0.0, 0.0})).dim() == 2, "mink_diff: works with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { mink_diff_known(b); });
    checks_and_their_switch();
    return test::finish("ellipsoid mink_diff");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
