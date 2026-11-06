// test_ellipsoid_plus - Ellipsoid plus: a vector, points, one dimension and the enclosure of Halder
//
// CORA: contSet/@ellipsoid/plus.m
// PROPERTY: contSet/@ellipsoid/plus.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "ellipsoid_testing.h"

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

/// Hand-computed: a vector, a point, the sum in 1D, parallel ellipsoids (Halder is exact there).
void plus_known(const std::string &b) {
    const Ellipsoid E(Tensor({{4.0, 0.0}, {0.0, 1.0}}), column({1.0, 2.0}));
    // a vector shifts the center
    check(close(E.plus(column({1.0, 1.0})).q(), {2, 3}, 0) &&
              close(E.plus(column({1.0, 1.0})).Q(), E.Q(), 0),
          b + ": a vector");
    const Ellipsoid point(Tensor::zeros({2, 2}), column({1.0, 1.0}));
    check(E.plus(point).isequal(E.plus(column({1.0, 1.0}))) &&
              point.plus(E).isequal(E.plus(column({1.0, 1.0}))),
          b + ": a point");
    // one dimension adds the radii, balls the radii as well
    const Ellipsoid a(Tensor({{4.0}}), column({1.0})), c(Tensor({{9.0}}), column({2.0}));
    check(close(a.plus(c).Q(), {25.0}, 1e-13) && close(a.plus(c).q(), {3.0}, 0),
          b + ": 1D: (2 + 3)^2");
    const Ellipsoid A(Tensor::eye(2), column({1.0, 0.0})),
        B(Tensor::eye(2) * 4.0, column({0.0, 1.0}));
    check(close(A.plus(B).Q(), {9, 0, 0, 9}, 1e-9) && close(A.plus(B).q(), {1, 1}, 0),
          b + ": balls: (1 + 2)^2 I");
    check(Ellipsoid::empty(2).plus(E).dim() == 2 && E.plus(Ellipsoid::empty(2)).Q().numel() == 0,
          b + ": empty");
    check(plus(A, B).is<Ellipsoid>(), b + ": registry");
}

/// Stated property: the sum of a point of each is a point of the enclosure.
void plus_property(const std::string &b) {
    Rng rng(127);
    for (int k = 0; k < 5; ++k) {
        const Ellipsoid A = test::random_ellipsoid(rng, 3), B = test::random_ellipsoid(rng, 3);
        const Tensor sum = test::ellipsoid_points(A, rng, 30) + test::ellipsoid_points(B, rng, 30);
        check(test::all_inside(A.plus(B), sum, 1e-6), b + ": the sums are inside");
    }
}

/// The SDP modes and degenerate sums are refused, as are other dimensions; checks off works.
void checks_and_their_switch() {
    const Ellipsoid E(Tensor({{4.0, 0.0}, {0.0, 1.0}}), column({1.0, 2.0}));
    const Ellipsoid B(Tensor::eye(2));
    check(test::raises([&] { E.plus(B, "outer"); }, Error::not_supported), "plus: outer");
    check(test::raises([&] { E.plus(B, "inner"); }, Error::not_supported), "plus: inner");
    check(test::raises([&] { E.plus(B, "best"); }, Error::wrong_value), "plus: mode");
    check(test::raises([&] { E.plus(Ellipsoid(Tensor({{1.0, 1.0}, {1.0, 1.0}}))); },
                       Error::not_supported),
          "plus: degenerate");
    check(test::raises([&] { E.plus(Ellipsoid(Tensor::eye(3))); }, Error::dimension_mismatch),
          "plus: dimension");
    check(test::raises([&] { E.plus(column({1.0})); }, Error::dimension_mismatch), "plus: vector");
    ChecksScope off(false);
    check(E.plus(B).dim() == 2, "plus: works with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        plus_known(b);
        plus_property(b);
    });
    checks_and_their_switch();
    return test::finish("ellipsoid plus");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
