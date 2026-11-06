// test_con_zonotope_rand_point - ConZonotope::rand_point: random points of a constrained zonotope
//
// CORA: contSet/@conZonotope/randPoint_.m
// PROPERTY: contSet/@conZonotope/randPoint_.m
// CORA authors: Niklas Kochdumper, Severin Prenitzer, Adrian Kulmburg
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "con_zonotope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Stated property: the points of either type are in the set, with and without constraints.
void rand_point_property(const std::string &b) {
    Rng rng(75);
    for (int k = 0; k < 3; ++k) {
        const ConZonotope cz = test::random_con_zonotope(rng, 3, 5, k);
        for (const char *type : {"standard", "extreme"}) {
            const Tensor points = cz.rand_point(8, rng, type);
            check(points.shape() == Shape{3, 8} && cz.contains(points),
                  b + ": " + type + " points are inside");
        }
        check(cz.contains(Set(cz)->rand_point(4, rng)), b + ": through the interface");
    }
    check(ConZonotope::empty(2).rand_point(3, rng, "standard").shape() == Shape{2, 0},
          b + ": the empty set has no point");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(rand_point_property);
    Rng rng(1);
    check(
        test::raises([&] { test::hexagon().rand_point(-1, rng, "standard"); }, Error::wrong_value),
        "rand_point: a negative number");
    check(test::raises([&] { test::hexagon().rand_point(2, rng, "uniform"); }, Error::wrong_value),
          "rand_point: another type");
    ChecksScope off(false);
    check(test::hexagon().rand_point(2, rng, "standard").shape() == Shape{2, 2},
          "rand_point: the same with the checks off");
    return test::finish("con_zonotope rand_point");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
