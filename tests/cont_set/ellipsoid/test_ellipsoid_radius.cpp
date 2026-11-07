// test_ellipsoid_radius - Ellipsoid radius: the semi-axes from the eigenvalues of Q
//
// CORA: contSet/@ellipsoid/radius.m
// PROPERTY: contSet/@ellipsoid/radius.m
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

/// Hand-computed: the i largest semi-axes descending; the empty ellipsoid has none.
void radius_known(const std::string &b) {
    const Ellipsoid E(Tensor({{1.0, 0.0}, {0.0, 4.0}}), column({1.0, 2.0}));
    check(close(E.radius(), {2.0}, 1e-14) && close(E.radius(2), {2.0, 1.0}, 1e-14),
          b + ": semi-axes");
    check(Ellipsoid::empty(2).radius().shape() == Shape({2, 0}), b + ": empty");
}

/// Stated property: the boundary points are at the distance of the largest semi-axis at most.
void radius_property(const std::string &b) {
    Rng rng(105);
    for (int k = 0; k < 4; ++k) {
        const Ellipsoid E = test::random_ellipsoid(rng, 3);
        const Tensor P = test::ellipsoid_points(E, rng, 30, true) - E.q();
        check(P.square().sum(0).sqrt().max().item() <= E.radius().item() + 1e-9,
              b + ": inside the ball of radius()");
    }
}

/// An axis count outside 1..n is refused; with the checks off a valid call is the same.
void checks_and_their_switch() {
    const Ellipsoid E(Tensor::eye(2));
    check(test::raises([&] { E.radius(0); }, Error::wrong_value) &&
              test::raises([&] { E.radius(3); }, Error::wrong_value),
          "radius: i");
    ChecksScope off(false);
    check(close(E.radius(2), {1.0, 1.0}, 1e-14), "radius: works with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        radius_known(b);
        radius_property(b);
    });
    checks_and_their_switch();
    return test::finish("ellipsoid radius");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
