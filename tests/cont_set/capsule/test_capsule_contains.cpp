// test_capsule_contains - Capsule contains: points, capsules, zonotopes and boxes
//
// CORA: contSet/@capsule/contains_.m
// PROPERTY: contSet/@capsule/contains_.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "capsule_testing.h"

using namespace cora;
using test::check;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hand-computed points: on the axis, beside it, at an end, beyond an end, on the boundary.
void contains_points(const std::string &b) {
    const Capsule C(column({0.0, 0.0}), column({1.0, 0.0}), 0.5);
    check(C.contains(column({0.0, 0.0})) && C.contains(column({0.5, 0.4})), b + ": inside");
    check(C.contains(column({1.5, 0.0})) && C.contains(column({-1.5, 0.0})), b + ": the ends");
    check(C.contains(column({0.0, 0.5})), b + ": the boundary beside the axis");
    check(!C.contains(column({0.0, 0.6})) && !C.contains(column({1.6, 0.0})),
          b + ": outside, beside and beyond");
    check(!C.contains(column({1.4, 0.4})), b + ": outside near an end (distance 0.566)");
    check(C.contains(column({1.4, 0.3})), b + ": on the rounded end (distance 0.5)");
    check(C.contains(Tensor({{0.0, 1.0}, {0.0, 0.2}})) &&
              !C.contains(Tensor({{0.0, 1.0}, {0.0, 0.7}})),
          b + ": several points need all of them");
    check(!C.contains(column({0.0, 0.5001})) && C.contains(column({0.0, 0.5001}), 1e-3),
          b + ": the tolerance relaxes the distance");
    check(C.contains(Tensor::zeros({2, 0})), b + ": no points");
    // a ball (zero generator) and a three-dimensional capsule
    const Capsule ball(column({1.0, 1.0}), column({0.0, 0.0}), 1.0);
    check(ball.contains(column({1.0, 2.0})) && !ball.contains(column({2.0, 2.0})), b + ": a ball");
    const Capsule space(column({0.0, 0.0, 0.0}), column({0.0, 0.0, 2.0}), 1.0);
    check(space.contains(column({0.5, 0.0, 2.5})) && !space.contains(column({0.5, 0.5, 2.9})),
          b + ": three dimensions");
    check(!Capsule::empty(2).contains(column({0.0, 0.0})), b + ": the empty capsule holds none");
}

/// Capsules: the two ends of the inner axis with its radius must be inside the outer capsule.
void contains_capsules(const std::string &b) {
    const Capsule outer(column({0.0, 0.0}), column({2.0, 0.0}), 1.0);
    check(outer.contains(Capsule(column({0.5, 0.0}), column({1.0, 0.0}), 0.5)), b + ": inside");
    check(outer.contains(Capsule(column({0.0, 0.0}), column({2.0, 0.0}), 1.0)), b + ": itself");
    check(!outer.contains(Capsule(column({0.0, 0.0}), column({2.0, 0.0}), 1.1)), b + ": too thick");
    check(!outer.contains(Capsule(column({2.0, 0.0}), column({1.0, 0.0}), 0.5)), b + ": too long");
    check(outer.contains(Capsule(column({0.0, 0.0}), column({0.0, 0.0}), 0.5)), b + ": a ball");
    check(outer.contains(Capsule::empty(2)), b + ": the empty capsule is inside");
    check(!Capsule::empty(2).contains(outer) && Capsule::empty(2).contains(Capsule::empty(2)),
          b + ": the empty capsule holds only itself");
    // through the registry and the empty set and the whole space
    check(contains(outer, Capsule(column({0.5, 0.0}), column({1.0, 0.0}), 0.5)),
          b + ": the free function");
    check(contains(outer, EmptySet(2)) && !contains(outer, Fullspace(2)), b + ": empty set, space");
    check(contains(EmptySet(2), Capsule::empty(2)) && !contains(EmptySet(2), outer),
          b + ": the empty set holds the empty capsule");
    check(contains(Fullspace(2), outer), b + ": the whole space holds it");
}

/// Zonotopes and boxes are inside exactly if their corners are.
void contains_sets(const std::string &b) {
    const Capsule C(column({0.0, 0.0}), column({2.0, 0.0}), 1.0);
    const Zonotope Z(column({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 0.5}}));
    check(C.contains(Z) && !C.contains(Zonotope(column({0.0, 0.0}), Tensor({{3.5}, {0.0}}))),
          b + ": a zonotope");
    check(C.contains(Interval(column({-1.0, -0.5}), column({1.0, 0.5}))) &&
              !C.contains(Interval(column({-1.0, -1.5}), column({1.0, 0.5}))),
          b + ": a box");
    check(C.contains(Interval::empty(2)), b + ": the empty box");
    check(contains(C, Z) && contains(C, Interval(column({0.0, 0.0}), column({0.1, 0.1}))),
          b + ": the free function");
}

/// A batch of capsules contains a point if every member does.
void contains_batch(const std::string &b) {
    const Tensor c = Tensor::from_data({0, 0, 5, 0}, {2, 2, 1});
    const Capsule batch(c, Tensor::zeros({2, 2, 1}), 1.0);
    check(!batch.contains(column({0.0, 0.0})), b + ": one member misses the point");
    check(batch.contains(Tensor::from_data({0.5, 0.0, 5.5, 0.0}, {2, 2, 1})),
          b + ": a point per member");
}

/// Stated property: sampled points of a capsule are inside, points beyond the radius are not
/// (against a distance found by brute force on a grid).
void contains_property(const std::string &b) {
    Rng rng(31);
    for (int k = 0; k < 8; ++k) {
        const Capsule C = test::random_capsule(rng, 3);
        check(C.contains(test::sample_points(C, rng, 40)), b + ": sampled points are inside");
        std::vector<double> data(3 * 30);
        rng.normal(data.data(), data.size(), 1.5);
        const double r = C.r().data()[0];
        for (int j = 0; j < 30; ++j) {
            const std::vector<double> p{data[j], data[30 + j], data[60 + j]};
            const double distance = test::axis_distance(C, p);
            if (std::abs(distance - r) < 1e-3) continue;
            check(C.contains(column(p)) == (distance < r), b + ": agrees with the grid distance");
        }
    }
}

/// Wrong dimensions, a negative tolerance, a zonotope of too many corners are refused; with the
/// checks off a valid call gives the same answer.
void checks_and_their_switch() {
    const Capsule C(column({0.0, 0.0}), column({1.0, 0.0}), 0.5);
    check(test::raises([&] { C.contains(column({0.0, 0.0, 0.0})); }, Error::dimension_mismatch),
          "contains: points of another dimension");
    check(test::raises([&] { C.contains(column({0.0, 0.0}), -1.0); }, Error::wrong_value),
          "contains: tol is nonnegative");
    check(test::raises([&] { C.contains(Capsule(column({1.0}))); }, Error::dimension_mismatch),
          "contains: a capsule of another dimension");
    check(test::raises([&] { C.contains(Zonotope(column({0.0}), Tensor({{1.0}}))); },
                       Error::dimension_mismatch),
          "contains: a zonotope of another dimension");
    check(
        test::raises([&] { C.contains(Zonotope(column({0.0, 0.0}), Tensor::eye(2).tile({1, 7}))); },
                     Error::not_supported),
        "contains: a zonotope of more than 12 generators");
    check(test::raises([&] { C.contains(Interval(column({0.0}), column({1.0}))); },
                       Error::dimension_mismatch),
          "contains: a box of another dimension");
    ChecksScope off(false);
    // the same with the checks off
    check(C.contains(column({0.0, 0.2})) && !C.contains(column({0.0, 2.0})),
          "contains: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        contains_points(b);
        contains_capsules(b);
        contains_sets(b);
        contains_batch(b);
        contains_property(b);
    });
    checks_and_their_switch();
    return test::finish("capsule contains");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
