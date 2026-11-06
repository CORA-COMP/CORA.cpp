// test_ellipsoid_distance - Ellipsoid distance: the measure of points to the ellipsoid
//
// CORA: contSet/@ellipsoid/distance.m
// PROPERTY: contSet/@ellipsoid/distance.m
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

/// Hand-computed: -1 at the center, 0 on the boundary, positive outside; the degenerate direction.
void distance_known(const std::string &b) {
    const Ellipsoid E(Tensor({{4.0, 0.0}, {0.0, 1.0}}), column({1.0, 2.0}));
    const Tensor p = Tensor({{1.0, 3.0, 1.0}, {2.0, 2.0, 4.0}});
    check(close(E.distance(p), {-1.0, 0.0, 3.0}, 1e-13), b + ": -1, 0, 3");
    const Ellipsoid flat(Tensor({{1.0, 0.0}, {0.0, 0.0}}), column({0.0, 0.0}), 1e-6);
    check(close(flat.distance(Tensor({{0.5}, {1e-3}})), {0.25 + 1.0 - 1.0}, 1e-8),
          b + ": a null direction counts 1 / tol");
}

/// Stated property: boundary points have distance 0, and inside it is not positive.
void distance_property(const std::string &b) {
    Rng rng(109);
    for (int k = 0; k < 4; ++k) {
        const Ellipsoid E = test::random_ellipsoid(rng, 3);
        check(close(E.distance(test::ellipsoid_points(E, rng, 20, true)), Tensor::zeros({1, 20}),
                    1e-8),
              b + ": 0 on the boundary");
        const Tensor inside = test::ellipsoid_points(E, rng, 20);
        check(E.distance(inside).max().item() <= 1e-9, b + ": not positive inside");
    }
}

/// A wrong number of rows, a batch and the empty ellipsoid are refused; checks off works.
void checks_and_their_switch() {
    const Ellipsoid E(Tensor({{4.0, 0.0}, {0.0, 1.0}}), column({1.0, 2.0}));
    check(test::raises([&] { E.distance(column({1.0})); }, Error::dimension_mismatch),
          "distance: dimension");
    check(test::raises([&] { Ellipsoid::empty(2).distance(column({1.0, 2.0})); }, Error::empty_set),
          "distance: empty");
    ChecksScope off(false);
    check(close(E.distance(column({1.0, 2.0})), {-1.0}, 1e-13),
          "distance: works with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        distance_known(b);
        distance_property(b);
    });
    checks_and_their_switch();
    return test::finish("ellipsoid distance");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
