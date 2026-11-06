// test_ellipsoid_is_intersecting - Ellipsoid is_intersecting: meeting points and ellipsoids
//
// CORA: contSet/@ellipsoid/isIntersecting_.m
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

/// Points are met if all lie in the ellipsoid; two ellipsoids where one is a point; the rest
/// refused.
void intersecting_known(const std::string &b) {
    const Ellipsoid E(Tensor({{4.0, 0.0}, {0.0, 1.0}}), column({1.0, 2.0}));
    check(E.is_intersecting(column({1.0, 2.0})) && !E.is_intersecting(column({5.0, 5.0})),
          b + ": points");
    const Ellipsoid point(Tensor::zeros({2, 2}), column({1.0, 3.0}));
    check(E.is_intersecting(point) && point.is_intersecting(E),
          b + ": a point on the boundary, either way");
    check(!E.is_intersecting(Ellipsoid(Tensor::zeros({2, 2}), column({9.0, 9.0}))),
          b + ": a point outside");
    check(!E.is_intersecting(Ellipsoid::empty(2)) && !Ellipsoid::empty(2).is_intersecting(E),
          b + ": empty");
    check(test::raises([&] { E.is_intersecting(E); }, Error::not_supported),
          b + ": two proper ones");
    check(is_intersecting(E, point) && is_intersecting(point, E), b + ": registry");
}

/// A wrong dimension is refused; with the checks off a valid call is the same.
void checks_and_their_switch() {
    const Ellipsoid E(Tensor({{4.0, 0.0}, {0.0, 1.0}}), column({1.0, 2.0}));
    check(test::raises([&] { E.is_intersecting(Ellipsoid(Tensor::eye(3))); },
                       Error::dimension_mismatch),
          "is_intersecting: dimension");
    check(test::raises([&] { E.is_intersecting(column({1.0})); }, Error::dimension_mismatch),
          "is_intersecting: points");
    check(test::raises([&] { E.is_intersecting(column({1.0, 2.0}), -1.0); }, Error::wrong_value),
          "is_intersecting: tol");
    ChecksScope off(false);
    check(E.is_intersecting(column({1.0, 2.0})), "is_intersecting: works with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { intersecting_known(b); });
    checks_and_their_switch();
    return test::finish("ellipsoid is_intersecting");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
