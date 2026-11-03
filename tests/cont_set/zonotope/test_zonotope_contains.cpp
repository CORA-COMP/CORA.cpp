// test_zonotope_contains - zonotope contains: a point, a zonotope or a box in a zonotope
//
// CORA: contSet/@zonotope/contains_.m
// PROPERTY: contSet/@zonotope/contains_.m

#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"
#include "testing.h"

using namespace cora;
using test::check;
using test::column;

namespace {

/// The diamond |x| + |y| <= 2.
Zonotope diamond() { return Zonotope(column({0.0, 0.0}), Tensor({{1.0, 1.0}, {1.0, -1.0}})); }

/// Points: inside, on the boundary, outside, and the tolerance that relaxes the boundary.
void points_in_a_diamond(const std::string &b) {
    const Zonotope Z = diamond();
    check(Z.contains(column({0.0, 0.0})), b + ": the center");
    check(Z.contains(column({1.0, 1.0})), b + ": the boundary");
    check(Z.contains(Tensor({{0.5, 2.0, -1.0}, {0.5, 0.0, 0.5}})), b + ": three points, all in");
    check(!Z.contains(Tensor({{0.5, 1.5}, {0.5, 1.5}})), b + ": one of two points is out");
    check(!Z.contains(column({1.0 + 1e-10, 1.0})), b + ": just outside, default tol");
    check(Z.contains(column({1.0 + 1e-10, 1.0}), "exact", 1e-6), b + ": just outside, tol 1e-6");
    check(Z.contains(column({1.0, 1.0}), "exact:venum") && Z.contains(column({1.0, 1.0}), "approx"),
          b + ": the other point methods");
}

/// The special cases of CORA's contains_: a point, a box, and a flat zonotope.
void special_zonotopes(const std::string &b) {
    const Zonotope point(column({1.0, 2.0}), Tensor({{0.0}, {0.0}}));
    check(point.contains(column({1.0, 2.0})) && !point.contains(column({1.0, 2.5})), b + ": a point");
    check(point.contains(Zonotope(column({1.0, 2.0}), Tensor({{0.0}, {0.0}}))), b + ": a point holds itself");
    check(!point.contains(diamond()), b + ": a point holds no set");
    const Zonotope box(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 1.0}}));
    check(box.contains(column({1.0, -1.0})) && !box.contains(column({1.1, 0.0})), b + ": a box");
    check(box.contains(Interval(column({-0.5, -0.5}), column({0.5, 1.0}))), b + ": a box holds a box");
    check(!box.contains(Interval(column({-0.5, -0.5}), column({0.5, 1.5}))), b + ": a box, not a larger one");
    const Zonotope segment(column({0.0, 0.0}), Tensor({{1.0}, {1.0}}));
    check(segment.contains(column({0.5, 0.5})), b + ": a segment holds its points");
    check(!segment.contains(column({0.5, 0.6})), b + ": and no other");
}

/// Sets: a zonotope in a zonotope by the vertex search and by Sadraddini and Tedrake.
void sets_in_the_diamond(const std::string &b) {
    const Zonotope Z = diamond();
    const Zonotope small = 0.5 * Z, large = 1.2 * Z;
    for (const std::string method : {"exact", "exact:venum", "approx", "approx:st"}) {
        check(Z.contains(small, method), b + ": a half in the diamond, " + method);
        check(!Z.contains(large, method), b + ": 1.2 times is not in, " + method);
    }
    check(Z.contains(Z), b + ": a zonotope holds itself");
    check(Z.contains(Interval(column({-0.5, -0.5}), column({0.5, 0.5}))), b + ": a box inside");
    check(!Z.contains(Interval(column({-1.5, -1.5}), column({1.5, 1.5}))), b + ": a box too large");
    check(Z.contains(small + column({0.5, 0.0})) && !Z.contains(small + column({1.5, 0.0})),
          b + ": shifted");
    check(contains(Z, small) && !contains(Z, large), b + ": through the registry");
    check(contains(Z, Interval(column({-0.5, -0.5}), column({0.5, 0.5}))),
          b + ": a box through the registry");
}

/// Stated property: c + s (Z' - c) is in Z for s < 1 and not for s > 1, for random zonotopes, and
/// every sampled point of Z is in Z.
void scaled_copies(const std::string &b) {
    cora::Rng rng(26);
    for (int trial = 0; trial < 6; ++trial) {
        const Zonotope Z = Zonotope::generate_random(3, 5, rng);
        const Tensor c = Z.center();
        const Zonotope inside(c, Z.generators() * 0.9), outside(c, Z.generators() * 1.1);
        for (const std::string method : {"exact", "approx"}) {
            check(Z.contains(inside, method), b + ": 0.9 Z in Z, " + method);
            check(!Z.contains(outside, method), b + ": 1.1 Z not in Z, " + method);
        }
        check(Z.contains(Z.rand_point(20, rng)), b + ": sampled points are in");
        check(Z.contains(Z.rand_point(20, rng, "extreme")), b + ": extreme points are in");
    }
}

/// Wrong operands and methods are refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const Zonotope Z = diamond();
    check(test::raises([&] { Z.contains(column({1.0, 2.0, 3.0})); }, Error::dimension_mismatch),
          "contains: a point of dimension 3");
    check(test::raises([&] { Z.contains(Zonotope(column({0.0}), Tensor({{1.0}}))); },
                       Error::dimension_mismatch), "contains: a zonotope of dimension 1");
    check(test::raises([&] { Z.contains(column({0.0, 0.0}), "exact", -1.0); }, Error::wrong_value),
          "contains: tol < 0");
    check(test::raises([&] { Z.contains(column({0.0, 0.0}), "simplex"); }, Error::no_specific_alg),
          "contains: unknown method");
    check(test::raises([&] { Z.contains(Z, "exact:polymax"); }, Error::not_supported),
          "contains: polymax needs Polytope");
    check(test::raises([&] { Z.contains(Z, "sampling"); }, Error::not_supported), "contains: sampling");
    check(test::raises([&] { Zonotope::stack({Z, Z}).contains(Z); }, Error::not_supported),
          "contains: a batch");
    const bool on = Z.contains(0.5 * Z);
    ChecksScope off(false);
    check(Z.contains(0.5 * Z) == on, "contains: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        points_in_a_diamond(b);
        special_zonotopes(b);
        sets_in_the_diamond(b);
        scaled_copies(b);
    });
    checks_and_their_switch();
    return test::finish("zonotope contains");
}
