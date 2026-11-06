// test_zono_bundle_is_intersecting - ZonoBundle::is_intersecting: whether it meets a set or points
//
// CORA: contSet/@zonoBundle/isIntersecting_.m
// CORA authors: Niklas Kochdumper
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

/// The exact test decides: sets that meet the intersection, sets that meet one member only, and
/// sets that touch it.
void exact_test(const std::string &b) {
    const ZonoBundle bundle = squares();
    const Tensor I2({{0.5, 0.0}, {0.0, 0.5}});
    const Zonotope inside(column({0.5, 0.5}), I2), in_one(column({-0.75, -0.75}), I2),
        touching(column({1.5, 0.5}), I2), apart(column({5.0, 5.0}), I2);
    check(bundle.is_intersecting(inside), b + ": a zonotope inside");
    check(!bundle.is_intersecting(in_one), b + ": a zonotope that meets one member only");
    check(bundle.is_intersecting(touching), b + ": a zonotope that touches the border");
    check(!bundle.is_intersecting(apart), b + ": a far zonotope");
    check(bundle.is_intersecting(Interval(column({0.9, 0.9}), column({3.0, 3.0}))), b + ": a box");
    check(!bundle.is_intersecting(Interval(column({-2.0, -2.0}), column({-0.5, -0.5}))),
          b + ": a box in the first member only");
    check(!bundle.is_intersecting(Interval::empty(2)), b + ": the empty box meets nothing");
    check(bundle.is_intersecting(ZonoBundle({inside, inside})), b + ": a bundle");
    check(!bundle.is_intersecting(ZonoBundle({in_one})), b + ": a bundle that misses");
    check(bundle.is_intersecting(column({0.5, 0.5})) &&
              !bundle.is_intersecting(column({-0.5, -0.5})),
          b + ": a point");
    check(!ZonoBundle::empty(2).is_intersecting(bundle), b + ": the empty bundle meets nothing");
    // the empty bundle and the registry
    check(is_intersecting(Set(bundle), Set(inside)) && is_intersecting(Set(inside), Set(bundle)),
          b + ": through the registry, in both orders");
}

/// The approximate test compares the boxes: it says true for sets that are apart but whose boxes
/// overlap, and false when the boxes are apart.
void approximate_test(const std::string &b) {
    const Tensor I2({{5.0, 0.0}, {0.0, 5.0}});
    const Zonotope diagonal(column({0.0, 0.0}), Tensor({{1.0}, {1.0}}));
    const ZonoBundle bundle({diagonal, Zonotope(column({0.0, 0.0}), I2)});
    const Zonotope off_diagonal(column({0.9, -0.9}), Tensor({{0.01}, {0.0}}));
    check(!bundle.is_intersecting(off_diagonal), b + ": exact: they are apart");
    check(bundle.is_intersecting(off_diagonal, "approx"), b + ": approx: the boxes overlap");
    check(!bundle.is_intersecting(Zonotope(column({9.0, 9.0}), Tensor({{1.0}, {1.0}})), "approx"),
          b + ": approx: the boxes are apart");
    check(
        bundle.is_intersecting(Interval(column({1.0001, 0.0}), column({2.0, 1.0})), "approx", 0.01),
        b + ": approx: the box enlarged by tol");
    check(
        !bundle.is_intersecting(Interval(column({1.0001, 0.0}), column({2.0, 1.0})), "approx", 0.0),
        b + ": approx: not without it");
    check(bundle.is_intersecting(ZonoBundle({off_diagonal}), "approx"), b + ": approx: a bundle");
}

/// A type that is neither, a batch and sets of another size are refused; the same call with the
/// checks off.
void checks_and_their_switch() {
    const ZonoBundle bundle = squares();
    const Zonotope Z(column({0.5, 0.5}), Tensor({{0.25}, {0.0}}));
    check(test::raises([&] { bundle.is_intersecting(Z, "fast"); }, Error::wrong_value),
          "is_intersecting: type");
    check(test::raises([&] { bundle.is_intersecting(Zonotope(column({0.5}), Tensor({{1.0}}))); },
                       Error::dimension_mismatch),
          "is_intersecting: another dimension");
    check(test::raises([&] { bundle.is_intersecting(Zonotope::stack({Z, Z})); },
                       Error::dimension_mismatch),
          "is_intersecting: a batch operand");
    check(test::raises([&] { bundle.is_intersecting(Z, "exact", -1.0); }, Error::wrong_value),
          "is_intersecting: tol < 0");
    ChecksScope off(false);
    check(bundle.is_intersecting(Z), "is_intersecting: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        exact_test(b);
        approximate_test(b);
    });
    checks_and_their_switch();
    return test::finish("zono_bundle is_intersecting");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
