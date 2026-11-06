// test_ellipsoid_rand_point - Ellipsoid rand_point: points of the ellipsoid, or of its boundary
//
// CORA: contSet/@ellipsoid/randPoint_.m
// PROPERTY: contSet/@ellipsoid/randPoint_.m
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

/// Stated property: standard and uniform points lie in the ellipsoid, extreme ones on its boundary.
void rand_point_property(const std::string &b) {
    Rng rng(131);
    for (int k = 0; k < 4; ++k) {
        const Ellipsoid E = test::random_ellipsoid(rng, 3, k == 3 ? 2 : -1);
        check(test::all_inside(E, E.rand_point(30, rng), 1e-6), b + ": standard points are inside");
        check(test::all_inside(E, E.rand_point(30, rng, "uniform"), 1e-6),
              b + ": uniform points are inside");
        const Tensor P = E.rand_point(30, rng, "extreme");
        bool boundary = true;
        for (const double x : test::norms_of(E, P)) boundary = boundary && std::abs(x - 1.0) < 1e-6;
        check(P.shape() == Shape({3, 30}) && boundary, b + ": extreme points are on the boundary");
    }
    const Ellipsoid point(Tensor::zeros({2, 2}), column({1.0, 2.0}));
    check(close(point.rand_point(3, rng), {1, 1, 1, 2, 2, 2}, 0),
          b + ": a point repeats its center");
}

/// Stated property: the volume share is uniform, here in 2D the share inside a smaller concentric
/// ellipsoid is its volume ratio.
void rand_point_uniform(const std::string &b) {
    Rng rng(137);
    const Ellipsoid E(Tensor::eye(2) * 4.0);
    const Tensor P = E.rand_point(4000, rng);
    // the unit disk of radius 1 inside the disk of radius 2 has a quarter of the area
    const double share =
        P.square().sum(0).sqrt().less_equal(P.square().sum(0).sqrt().ones_like()).sum().item() /
        4000.0;
    check(std::abs(share - 0.25) < 0.04, b + ": uniform in the volume");
}

/// An unknown type, a negative number, the empty ellipsoid and a batch are refused; checks off
/// works.
void checks_and_their_switch() {
    Rng rng(3);
    const Ellipsoid E(Tensor::eye(2));
    check(test::raises([&] { E.rand_point(3, rng, "gaussian"); }, Error::no_specific_alg),
          "rand_point: type");
    check(test::raises([&] { E.rand_point(-1, rng); }, Error::wrong_value), "rand_point: N");
    check(test::raises([&] { Ellipsoid::empty(2).rand_point(1, rng); }, Error::empty_set),
          "rand_point: empty");
    const Ellipsoid batch(Tensor::zeros({2, 2, 2}), Tensor::zeros({2, 2, 1}));
    check(test::raises([&] { batch.rand_point(1, rng); }, Error::not_supported),
          "rand_point: batch");
    ChecksScope off(false);
    check(E.rand_point(2, rng).shape() == Shape({2, 2}), "rand_point: works with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        rand_point_property(b);
        rand_point_uniform(b);
    });
    checks_and_their_switch();
    return test::finish("ellipsoid rand_point");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
