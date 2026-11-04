// test_zono_bundle_con_zonotope - ZonoBundle::con_zonotope: the bundle as a constrained zonotope
//
// CORA: contSet/@zonoBundle/conZonotope.m
// PROPERTY: contSet/@zonoBundle/conZonotope.m
//
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "zono_bundle_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Two squares [0, 2]^2 and [1, 3]^2 meet in [1, 2]^2: one equality per dimension joins them.
void two_squares(const std::string &b) {
    const Tensor I2({{1.0, 0.0}, {0.0, 1.0}});
    const ZonoBundle bundle({Zonotope(column({1.0, 1.0}), I2), Zonotope(column({2.0, 2.0}), I2)});
    const ConZonotope cz = bundle.con_zonotope();
    check(cz.generators().shape() == Shape{2, 4}, b + ": the generators of both members");
    check(cz.A().shape() == Shape{2, 4}, b + ": one equality per dimension");
    const Interval box = cz.interval();
    check(close(box.inf(), {1.0, 1.0}) && close(box.sup(), {2.0, 2.0}), b + ": the box [1, 2]^2");
    check(cz.contains(column({1.5, 1.5})) && !cz.contains(column({0.5, 1.5})),
          b + ": a point inside and one outside");
}

/// One member is the zonotope itself, without constraints.
void single_member(const std::string &b) {
    const Zonotope Z(column({1.0, 2.0}), Tensor({{1.0, 0.5}, {0.0, 1.0}}));
    const ConZonotope cz = ZonoBundle({Z}).con_zonotope();
    check(cz.A().shape() == Shape{0, 2} && close(cz.c(), Z.center(), 0) &&
              close(cz.generators(), Z.generators(), 0),
          b + ": a single member has no constraints");
}

/// Stated property: the points of the bundle are the points of the constrained zonotope.
void same_points(const std::string &b) {
    Rng rng(21);
    Tensor common;
    const ZonoBundle bundle = test::random_zono_bundle(rng, 3, 3, 4, &common);
    const ConZonotope cz = bundle.con_zonotope();
    const Tensor inside = test::bundle_points(bundle, rng, 12, common);
    check(cz.contains(inside), b + ": every point of the bundle is in the constrained zonotope");
    // a point of the first member only: the others cut it off
    const Tensor far =
        bundle.zonotopes()[0].center() + bundle.zonotopes()[0].generators().slice(-1, 0, 1) * 1.0;
    check(cz.contains(far) == test::in_every_member(bundle, far), b + ": a boundary point agrees");
    check(!cz.contains(common + common.ones_like() * 100.0), b + ": a far point is outside");
}

/// The disjoint members give an empty set; the call takes no input and runs with the checks off.
void empty_bundle_and_switch() {
    const ConZonotope cz = ZonoBundle::empty(2).con_zonotope();
    check(cz.interval().is_empty(), "con_zonotope: the empty bundle is empty");
    ChecksScope off(false);
    check(ZonoBundle::empty(2).con_zonotope().interval().is_empty(),
          "con_zonotope: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        two_squares(b);
        single_member(b);
        same_points(b);
    });
    empty_bundle_and_switch();
    return test::finish("zono_bundle con_zonotope");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
