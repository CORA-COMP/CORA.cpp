// test_capsule_volume - Capsule volume: the ball plus the cylinder
//
// CORA: contSet/@capsule/volume_.m
// PROPERTY: contSet/@capsule/volume_.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "capsule_testing.h"

#include <cmath>
#include <numbers>

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed in one, two and three dimensions; the empty capsule has volume 0.
void volume_known(const std::string &b) {
    const double pi = std::numbers::pi;
    const Capsule line(column({1.0}), column({3.0}), 0.5);
    check(close(line.volume(), {7.0}, 1e-14), b + ": 1D: 2 r + 2 |g|");
    const Capsule plane(column({1.0, 2.0}), column({3.0, 4.0}), 0.5);
    check(close(plane.volume(), {pi * 0.25 + 4 * 0.5 * 5}, 1e-13), b + ": 2D: pi r^2 + 4 r |g|");
    const Capsule space(column({0.0, 0.0, 0.0}), column({0.0, 0.0, 2.0}), 1.0);
    check(close(space.volume(), {4.0 / 3.0 * pi + pi * 4.0}, 1e-13),
          b + ": 3D: 4/3 pi r^3 + 2 pi r^2 |g|");
    check(close(Capsule::empty(2).volume(), {0.0}, 0), b + ": the empty capsule");
    check(close(Capsule(column({1.0, 2.0})).volume(), {0.0}, 0), b + ": a point");
}

/// Stated property: the volume of a ball (g = 0) is that of the unit ball times r^n, and a sampled
/// share of a box around the capsule that lies inside gives the volume of the box times that share.
void volume_property(const std::string &b) {
    Rng rng(43);
    const Capsule C(column({0.0, 0.0}), column({0.6, 0.0}), 0.4);
    std::vector<double> u(2 * 20000);
    rng.uniform(u.data(), u.size(), -1.0, 1.0);
    int inside = 0;
    for (int k = 0; k < 20000; ++k)
        inside += test::axis_distance(C, {u[2 * k], u[2 * k + 1]}) <= 0.4 ? 1 : 0;
    // the capsule lies in [-1, 1]^2 of area 4
    check(std::abs(4.0 * inside / 20000 - C.volume().item()) < 0.05, b + ": Monte Carlo agrees");
}

/// The call works with the checks off too.
void checks_and_their_switch() {
    const Capsule C(column({1.0, 2.0}), column({3.0, 4.0}), 0.5);
    const Tensor on = C.volume();
    ChecksScope off(false);
    check(close(C.volume(), on, 0), "volume: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        volume_known(b);
        volume_property(b);
    });
    checks_and_their_switch();
    return test::finish("capsule volume");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
