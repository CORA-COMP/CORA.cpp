// test_zono_bundle_contains - ZonoBundle::contains: points, a zonotope, a box or a bundle
//
// CORA: contSet/@zonoBundle/contains_.m
// PROPERTY: contSet/@zonoBundle/contains_.m
//
// CORA authors: Niklas Kochdumper, Mark Wetzlinger, Adrian Kulmburg
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/registry.h"
#include "zono_bundle_testing.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The squares [-1, 1]^2 and [0, 2]^2 meet in [0, 1]^2.
ZonoBundle squares() {
    const Tensor I2({{1.0, 0.0}, {0.0, 1.0}});
    return ZonoBundle({Zonotope(column({0.0, 0.0}), I2), Zonotope(column({1.0, 1.0}), I2)});
}

/// Points: inside, in one member only and outside, by both exact methods ("approx" is "exact").
void points(const std::string &b) {
    const ZonoBundle zB = squares();
    for (const std::string method : {"exact", "exact:zonotope", "approx"}) {
        check(zB.contains(column({0.5, 0.5}), method), b + ": inside, " + method);
        check(!zB.contains(column({-0.5, -0.5}), method), b + ": in one member only, " + method);
        check(!zB.contains(column({5.0, 0.0}), method), b + ": outside, " + method);
    }
    const Tensor two_points = Tensor({{0.5, 0.25}, {0.5, 0.75}});
    check(zB.contains(two_points), b + ": all columns must be inside");
    check(!zB.contains(Tensor({{0.5, 5.0}, {0.5, 5.0}})), b + ": one column outside");
    check(zB.contains(column({1.0 + 1e-15, 0.5})), b + ": a point on the border, up to the tolerance");
    check(!zB.contains(column({1.001, 0.5}), "exact", 1e-6), b + ": beyond the tolerance");
}

/// A zonotope or a box is inside if every member holds it; a bundle if each facet of each member
/// holds it.
void sets(const std::string &b) {
    const ZonoBundle zB = squares();
    const Zonotope small(column({0.5, 0.5}), Tensor({{0.25, 0.0}, {0.0, 0.25}})),
        shifted(column({0.0, 0.0}), Tensor({{0.5, 0.0}, {0.0, 0.5}}));
    check(zB.contains(small), b + ": a small zonotope");
    check(!zB.contains(shifted), b + ": a zonotope that leaves the second member");
    check(zB.contains(Interval(column({0.25, 0.25}), column({0.75, 0.75}))), b + ": a box inside");
    check(!zB.contains(Interval(column({-0.25, 0.25}), column({0.75, 0.75}))), b + ": a box outside");
    check(zB.contains(Interval::empty(2)), b + ": the empty box");
    const ZonoBundle inner = zB.intersect(small);
    check(zB.contains(inner) && !inner.contains(zB) && zB.contains(zB), b + ": bundles");
    check(contains(Set(zB), Set(small)) && !contains(Set(zB), Set(shifted)),
          b + ": through the registry");
    check(zB.contains(ZonoBundle::empty(2)), b + ": the empty bundle is inside");
    check(!ZonoBundle::empty(2).contains(zB), b + ": the empty bundle holds nothing");
}

/// Stated property: sampled points of the bundle are inside, and a bundle holds the intersection
/// with anything.
void sampled_points(const std::string &b) {
    Rng rng(31);
    Tensor x0, y0;
    const ZonoBundle zB = test::random_zono_bundle(rng, 3, 3, 4, &x0);
    const Tensor points = test::bundle_points(zB, rng, 10, x0);
    check(zB.contains(points) && zB.contains(points, "exact:zonotope"), b + ": sampled points");
    const Zonotope other = test::random_zono_bundle(rng, 3, 1, 4, &y0).zonotopes()[0];
    check(zB.contains(zB.intersect(other)), b + ": the intersection with a zonotope");
}

/// Unknown and unavailable methods, a batch and points of another size are refused; the same call
/// with the checks off.
void checks_and_their_switch() {
    const ZonoBundle zB = squares();
    const Zonotope Z(column({0.5, 0.5}), Tensor({{0.25}, {0.0}}));
    check(test::raises([&] { zB.contains(column({0.5, 0.5}), "exact:polytope"); }, Error::not_supported),
          "contains: exact:polytope needs Polytope");
    check(test::raises([&] { zB.contains(Z, "approx"); }, Error::not_supported),
          "contains: approx for a set");
    check(test::raises([&] { zB.contains(column({0.5, 0.5}), "sampling"); }, Error::no_specific_alg),
          "contains: an unknown method");
    check(test::raises([&] { zB.contains(column({0.5})); }, Error::dimension_mismatch),
          "contains: points of another dimension");
    check(test::raises([&] { zB.contains(Zonotope::stack({Z, Z})); }, Error::dimension_mismatch),
          "contains: a batch operand");
    check(test::raises([&] { zB.contains(column({0.5, 0.5}), "exact", -1.0); }, Error::wrong_value),
          "contains: tol < 0");
    ChecksScope off(false);
    check(zB.contains(column({0.5, 0.5})), "contains: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        points(b);
        sets(b);
        sampled_points(b);
    });
    checks_and_their_switch();
    return test::finish("zono_bundle contains");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
