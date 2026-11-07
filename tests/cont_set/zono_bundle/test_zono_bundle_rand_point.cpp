// test_zono_bundle_rand_point - ZonoBundle::rand_point: random points of a bundle
//
// CORA: contSet/@zonoBundle/randPoint_.m
// PROPERTY: contSet/@zonoBundle/randPoint_.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "zono_bundle_testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Stated property: the points of either type are in every member.
void rand_point_property(const std::string &b) {
    Rng rng(69);
    for (int k = 0; k < 3; ++k) {
        const ZonoBundle zono_bundle = test::random_zono_bundle(rng, 3, 2, 4);
        for (const char *type : {"standard", "extreme"}) {
            const Tensor points = zono_bundle.rand_point(6, rng, type);
            check(points.shape() == Shape{3, 6} && zono_bundle.contains(points),
                  b + ": " + type + " points are inside");
        }
        check(zono_bundle.contains(Set(zono_bundle)->rand_point(4, rng)),
              b + ": through the interface");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(rand_point_property);
    Rng rng(1);
    const ZonoBundle zono_bundle = test::two_zonotopes();
    check(test::raises([&] { zono_bundle.rand_point(-1, rng, "standard"); }, Error::wrong_value),
          "rand_point: a negative number");
    check(test::raises([&] { zono_bundle.rand_point(2, rng, "uniform"); }, Error::wrong_value),
          "rand_point: another type");
    ChecksScope off(false);
    check(zono_bundle.rand_point(2, rng, "standard").shape() == Shape{2, 2},
          "rand_point: the same with the checks off");
    return test::finish("zono_bundle rand_point");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
