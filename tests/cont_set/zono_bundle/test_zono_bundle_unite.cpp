// test_zono_bundle_unite - ZonoBundle::unite: an enclosure of the union of a bundle with a set
//
// CORA: contSet/@zonoBundle/or.m
// PROPERTY: contSet/@zonoBundle/or.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "zono_bundle_testing.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Stated property: the union contains the points of both sets.
void unite_property(const std::string &b) {
    Rng rng(70);
    Tensor common;
    const ZonoBundle zono_bundle = test::random_zono_bundle(rng, 2, 2, 3, &common);
    const Tensor x = test::bundle_points(zono_bundle, rng, 10, common);
    const Zonotope Z = Zonotope::generate_random(rng, 2);
    const Interval I(column({5.0, 5.0}), column({6.0, 7.0}));
    Tensor other;
    const ZonoBundle second = test::random_zono_bundle(rng, 2, 2, 3, &other);
    const Tensor y = test::bundle_points(second, rng, 10, other);
    check(zono_bundle.unite(Z).contains(x) && zono_bundle.unite(Z).contains(Z.rand_point(8, rng)),
          b + ": a zonotope");
    check(zono_bundle.unite(I).contains(x) && zono_bundle.unite(I).contains(column({5.5, 6.0})),
          b + ": a box");
    check(zono_bundle.unite(second).contains(x) && zono_bundle.unite(second).contains(y),
          b + ": a bundle");
    check(zono_bundle.unite(Z).parallel_sets() == 1, b + ": the union has no constraint left");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(unite_property);
    const ZonoBundle zono_bundle = test::two_zonotopes();
    const Zonotope other(column({0.0}), Tensor({{1.0}}));
    check(test::raises([&] { zono_bundle.unite(other); }, Error::dimension_mismatch),
          "unite: a zonotope of another dimension");
    ChecksScope off(false);
    check(zono_bundle.unite(zono_bundle).dim() == 2, "unite: the same with the checks off");
    return test::finish("zono_bundle unite");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
