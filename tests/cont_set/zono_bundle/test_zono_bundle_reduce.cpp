// test_zono_bundle_reduce - ZonoBundle::reduce: every member with fewer generators
//
// CORA: contSet/@zonoBundle/reduce.m
// PROPERTY: contSet/@zonoBundle/reduce.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "zono_bundle_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The members have at most order * n generators, and the bundle stays an enclosure.
void reduce_property(const std::string &b) {
    Rng rng(65);
    Tensor common;
    const ZonoBundle zono_bundle = test::random_zono_bundle(rng, 2, 3, 7, &common);
    const ZonoBundle R = zono_bundle.reduce(2);
    bool few = R.parallel_sets() == zono_bundle.parallel_sets();
    for (const Zonotope &Z : R.zonotopes()) few = few && Z.generators().shape().back() <= 4;
    check(few, b + ": at most order * n generators per member");
    check(R.contains(test::bundle_points(zono_bundle, rng, 15, common)),
          b + ": the points are inside");
    check(Set(zono_bundle)->reduce(1).as<ZonoBundle>().zonotopes()[0].generators().shape().back() <=
              2,
          b + ": through the interface");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(reduce_property);
    const ZonoBundle zono_bundle = test::two_zonotopes();
    check(test::raises([&] { zono_bundle.reduce(0); }, Error::wrong_value), "reduce: the order 0");
    check(test::raises([&] { zono_bundle.reduce(1, "pca"); }, Error::not_supported),
          "reduce: another method");
    ChecksScope off(false);
    check(zono_bundle.reduce(1).parallel_sets() == 2, "reduce: the same with the checks off");
    return test::finish("zono_bundle reduce");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
