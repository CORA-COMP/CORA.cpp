// test_capsule_is_intersecting - Capsule is_intersecting: whether two capsules meet
//
// CORA: contSet/@capsule/isIntersecting_.m
// PROPERTY: contSet/@capsule/isIntersecting_.m

#include "capsule_testing.h"

using namespace cora;
using test::check;
using test::column;

namespace {

/// Hand-computed: parallel and crossing segments, touching, apart, an end against a side.
void is_intersecting_known(const std::string &b) {
    const Capsule C1(column({0.0, 0.0}), column({1.0, 0.0}), 0.1);
    // CORA's quadratic program finds a minimizer at the box and says false here: the axes cross at
    // (0.5, 0), so the capsules overlap
    const Capsule cross(column({0.5, 0.0}), column({0.0, 1.0}), 0.1);
    check(C1.is_intersecting(cross), b + ": crossing axes (CORA says false)");
    const Capsule parallel(column({0.0, 1.0}), column({1.0, 0.0}), 0.4);
    check(!C1.is_intersecting(parallel), b + ": parallel at distance 1, radii 0.5");
    check(C1.is_intersecting(Capsule(column({0.0, 1.0}), column({1.0, 0.0}), 0.9)),
          b + ": parallel, radii 1.0: touching");
    check(!C1.is_intersecting(Capsule(column({3.0, 0.0}), column({0.0, 1.0}), 0.1)),
          b + ": beyond the end");
    check(C1.is_intersecting(Capsule(column({2.0, 0.0}), column({0.0, 1.0}), 0.95)),
          b + ": an end against a side");
    check(C1.is_intersecting(Capsule(column({0.0, 0.0}), column({0.0, 0.0}), 0.1)),
          b + ": a ball on the axis");
    // three dimensions: skew lines at distance 1
    const Capsule x_axis(column({0.0, 0.0, 0.0}), column({1.0, 0.0, 0.0}), 0.4);
    const Capsule skew(column({0.0, 0.0, 1.0}), column({0.0, 1.0, 0.0}), 0.4);
    check(!x_axis.is_intersecting(skew), b + ": skew segments, radii 0.8 < 1");
    check(x_axis.is_intersecting(Capsule(column({0.0, 0.0, 1.0}), column({0.0, 1.0, 0.0}), 0.6)),
          b + ": skew segments, radii 1.0");
    check(!C1.is_intersecting(Capsule::empty(2)) && !Capsule::empty(2).is_intersecting(C1),
          b + ": the empty capsule meets nothing");
    check(C1.is_intersecting(column({0.5, 0.05})) && !C1.is_intersecting(column({0.5, 0.5})),
          b + ": points are contains");
    // the registry and the empty set and the whole space
    check(is_intersecting(C1, cross) && !is_intersecting(C1, EmptySet(2)) &&
              is_intersecting(Fullspace(2), C1),
          b + ": the free function");
}

/// Stated property: the answer agrees with the distance of the axes found on a grid, and the
/// relation is symmetric.
void is_intersecting_property(const std::string &b) {
    Rng rng(37);
    for (int k = 0; k < 12; ++k) {
        const Capsule C1 = test::random_capsule(rng, 3), C2 = test::random_capsule(rng, 3);
        check(C1.is_intersecting(C2) == C2.is_intersecting(C1), b + ": symmetric");
        // the least distance of points of the second axis from the first, on a grid
        double least = 1e300;
        const std::vector<double> c = C2.c().data(), g = C2.generator().data();
        for (int j = 0; j <= 2000; ++j) {
            const double t = -1.0 + 2.0 * j / 2000.0;
            least = std::min(least, test::axis_distance(
                                        C1, {c[0] + t * g[0], c[1] + t * g[1], c[2] + t * g[2]}));
        }
        const double reach = C1.r().data()[0] + C2.r().data()[0];
        if (std::abs(least - reach) < 1e-2) continue;
        check(C1.is_intersecting(C2) == (least < reach), b + ": agrees with the grid distance");
    }
}

/// Capsules of different dimensions, a negative tolerance are refused; with the checks off the
/// same valid call gives the same answer.
void checks_and_their_switch() {
    const Capsule C1(column({0.0, 0.0}), column({1.0, 0.0}), 0.1);
    check(test::raises([&] { C1.is_intersecting(Capsule(column({1.0}))); },
                       Error::dimension_mismatch),
          "is_intersecting: the dimensions agree");
    check(test::raises([&] { C1.is_intersecting(C1, -1.0); }, Error::wrong_value),
          "is_intersecting: tol is nonnegative");
    check(test::raises([&] { C1.is_intersecting(column({0.0, 0.0, 0.0})); },
                       Error::dimension_mismatch),
          "is_intersecting: points of another dimension");
    ChecksScope off(false);
    check(C1.is_intersecting(C1), "is_intersecting: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        is_intersecting_known(b);
        is_intersecting_property(b);
    });
    checks_and_their_switch();
    return test::finish("capsule is_intersecting");
}
