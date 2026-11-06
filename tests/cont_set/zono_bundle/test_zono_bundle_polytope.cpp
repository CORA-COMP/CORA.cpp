// test_zono_bundle_polytope - ZonoBundle::polytope: the halfspace form of a bundle
//
// CORA: contSet/@zonoBundle/polytope.m
// PROPERTY: contSet/@zonoBundle/polytope.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/polytope/polytope.h"
#include "zono_bundle_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The intersection of the two zonotopes: points of both members are in, points outside either
/// member are out.
void polytope_known(const std::string &b) {
    const Polytope P = test::two_zonotopes().polytope();
    check(P.contains(column({0.0, 0.0})) && P.contains(column({2.0, 0.0})) &&
              P.contains(column({1.0, 1.0})),
          b + ": points of both members");
    check(!P.contains(column({2.5, 0.0})) && !P.contains(column({-1.5, 0.0})) &&
              !P.contains(column({0.0, 1.5})),
          b + ": points outside a member");
}

/// Stated property: the points of the bundle are in the polytope.
void polytope_property(const std::string &b) {
    Rng rng(67);
    Tensor common;
    const ZonoBundle zono_bundle = test::random_zono_bundle(rng, 3, 2, 4, &common);
    check(zono_bundle.polytope().contains(test::bundle_points(zono_bundle, rng, 12, common)),
          b + ": the points are inside");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        polytope_known(b);
        polytope_property(b);
    });
    const Zonotope Z(column({0.0}), Tensor({{1.0}}));
    check(test::raises([&] { ZonoBundle({Zonotope::stack({Z, Z})}).polytope(); },
                       Error::not_supported),
          "polytope: a batch");
    ChecksScope off(false);
    check(test::two_zonotopes().polytope().dim() == 2, "polytope: the same with the checks off");
    return test::finish("zono_bundle polytope");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
