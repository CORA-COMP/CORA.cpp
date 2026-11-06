// test_ellipsoid_volume - Ellipsoid volume: the unit ball times sqrt(det Q)
//
// CORA: contSet/@ellipsoid/volume_.m
// PROPERTY: contSet/@ellipsoid/volume_.m
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

/// Hand-computed: 2D and 1D, a point, the empty ellipsoid.
void volume_known(const std::string &b) {
    const Ellipsoid E(Tensor({{1.0, 0.0}, {0.0, 4.0}}), column({1.0, 2.0}));
    check(close(E.volume(), {2.0 * std::numbers::pi}, 1e-13), b + ": pi * 2");
    check(close(Ellipsoid(Tensor({{9.0}})).volume(), {6.0}, 1e-14), b + ": 1D: 2 * 3");
    check(close(Ellipsoid(Tensor::zeros({2, 2})).volume(), {0.0}, 0), b + ": a point");
    check(close(Ellipsoid::empty(2).volume(), {0.0}, 0), b + ": empty");
}

/// Stated property: the volume of a ball of radius r is 4/3 pi r^3 in 3D, and it scales with
/// sqrt(det).
void volume_property(const std::string &b) {
    const Ellipsoid ball(Tensor::eye(3) * 4.0);
    check(close(ball.volume(), {4.0 / 3.0 * std::numbers::pi * 8.0}, 1e-12),
          b + ": 3D ball of radius 2");
    Rng rng(107);
    const Ellipsoid E = test::random_ellipsoid(rng, 2);
    check(close(Ellipsoid(E.Q() * 2.0).volume().item(), 2.0 * E.volume().item(), 1e-12),
          b + ": scaling");
}

/// The call works with the checks off too.
void checks_and_their_switch() {
    const Ellipsoid E(Tensor::eye(2));
    const Tensor on = E.volume();
    ChecksScope off(false);
    check(close(E.volume(), on, 0), "volume: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        volume_known(b);
        volume_property(b);
    });
    checks_and_their_switch();
    return test::finish("ellipsoid volume");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
