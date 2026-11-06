// test_zono_bundle_intersect - ZonoBundle::intersect: the members of both sets
//
// CORA: contSet/@zonoBundle/and_.m
// PROPERTY: contSet/@zonoBundle/and_.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/registry.h"
#include "zono_bundle_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A bundle, a zonotope and a box are appended in order; a box becomes its zonotope.
void members_are_appended(const std::string &b) {
    const Zonotope Z1(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}})),
        Z2(column({1.0, 0.0}), Tensor({{1.0}, {1.0}})),
        Z3(column({0.0, 1.0}), Tensor({{2.0}, {0.0}}));
    const ZonoBundle bundle({Z1, Z2});
    const ZonoBundle with_bundle = bundle.intersect(ZonoBundle({Z3, Z1}));
    check(with_bundle.parallel_sets() == 4 &&
              close(with_bundle.zonotopes()[2].center(), Z3.center(), 0),
          b + ": a bundle: the members of both");
    check(bundle.intersect(Z3).parallel_sets() == 3 &&
              close(bundle.intersect(Z3).zonotopes()[2].generators(), Z3.generators(), 0),
          b + ": a zonotope: one more member");
    const Interval I(column({-1.0, -1.0}), column({1.0, 3.0}));
    check(close(bundle.intersect(I).zonotopes()[2].center(), I.zonotope().center(), 0),
          b + ": a box: its zonotope");
    check(bundle.parallel_sets() == 2, b + ": the operand is unchanged");
}

/// The registry finds the pairs in either order, as CORA's precedence puts the bundle first.
void through_the_registry(const std::string &b) {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{1.0}, {1.0}}));
    const ZonoBundle bundle({Z, Z});
    const Set left = intersect(Set(bundle), Set(Z)), right = intersect(Set(Z), Set(bundle));
    check(left.is<ZonoBundle>() && left.as<ZonoBundle>().parallel_sets() == 3,
          b + ": bundle, zonotope");
    check(right.is<ZonoBundle>() && right.as<ZonoBundle>().parallel_sets() == 3,
          b + ": zonotope, bundle");
    check((Set(bundle) & Set(bundle)).as<ZonoBundle>().parallel_sets() == 4,
          b + ": the operator &");
}

/// Stated property: a point is in the result if and only if it is in the bundle and in the set.
void points_of_the_intersection(const std::string &b) {
    Rng rng(26);
    Tensor common, other;
    const ZonoBundle bundle = test::random_zono_bundle(rng, 3, 2, 3, &common);
    const Zonotope Z = test::random_zono_bundle(rng, 3, 1, 3, &other).zonotopes()[0];
    const ZonoBundle R = bundle.intersect(Z);
    const Tensor points = test::bundle_points(bundle, rng, 20, common);
    for (int k = 0; k < 20; ++k) {
        const Tensor p = points.slice(-1, k, k + 1);
        check(test::in_every_member(R, p) == (test::in_every_member(bundle, p) && Z.contains(p)),
              b + ": membership of a sampled point");
    }
}

/// A set of another dimension or batch shape and an empty box are refused; the same call with the
/// checks off.
void checks_and_their_switch() {
    const Zonotope Z(column({0.0, 0.0}), Tensor({{1.0}, {1.0}}));
    const ZonoBundle bundle({Z});
    check(test::raises([&] { bundle.intersect(Zonotope(column({0.0}), Tensor({{1.0}}))); },
                       Error::dimension_mismatch),
          "intersect: another dimension");
    check(
        test::raises([&] { bundle.intersect(Zonotope::stack({Z, Z})); }, Error::dimension_mismatch),
        "intersect: another batch shape");
    check(test::raises([&] { bundle.intersect(Interval::empty(2)); }, Error::not_supported),
          "intersect: an empty box");
    check(test::raises([&] { intersect(Set(bundle), Set(Interval::unbounded(3))); },
                       Error::dimension_mismatch),
          "intersect: the registry checks the dimension");
    ChecksScope off(false);
    check(bundle.intersect(Z).parallel_sets() == 2, "intersect: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        members_are_appended(b);
        through_the_registry(b);
        points_of_the_intersection(b);
    });
    checks_and_their_switch();
    return test::finish("zono_bundle intersect");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
