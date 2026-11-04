// test_capsule_convex_hull - Capsule convex_hull: a capsule around two capsules
//
// CORA: contSet/@capsule/convHull_.m
// PROPERTY: contSet/@capsule/convHull_.m

#include "capsule_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

namespace {

/// Hand-computed: the axis runs between the centers, the radius is the larger |g| + r.
void convex_hull_known(const std::string &b) {
    const Capsule C1(column({0.0, 0.0}), column({1.0, 0.0}), 0.5);
    const Capsule C2(column({4.0, 0.0}), column({0.0, 1.0}), 0.25);
    const Capsule H = C1.convex_hull(C2);
    // c_ = (2, 0), g_ = (c1 - c2)/2 = (-2, 0), r_ = max(1 + 0.5, 1 + 0.25)
    check(close(H.c(), {2, 0}, 0) && close(H.generator(), {-2, 0}, 0) && close(H.r(), {1.5}, 1e-15),
          b + ": center, generator and radius");
    check(close(C1.enclose(C2).c(), H.c(), 0), b + ": enclose is the hull");
    // the hull with an empty capsule is the other one, in either order
    check(C1.convex_hull(Capsule::empty(2)).isequal(C1, 0.0), b + ": with empty");
    check(Capsule::empty(2).convex_hull(C2).isequal(C2, 0.0), b + ": empty with");
    check(Capsule::empty(2).convex_hull(Capsule::empty(2)).is_empty(), b + ": two empty");
    // through the registry, also with the empty set and the whole space
    check(convex_hull(C1, C2).as<Capsule>().isequal(H, 0.0), b + ": the free function");
    check(convex_hull(C1, EmptySet(2)).as<Capsule>().isequal(C1, 0.0), b + ": with the empty set");
    check(convex_hull(Fullspace(2), C1).class_id() == ClassId::fullspace,
          b + ": with the whole space");
    // a batch hulls member by member
    const Tensor c = Tensor::from_data({0, 0, 4, 0}, {2, 2, 1});
    const Capsule batch(c, Tensor::zeros({2, 2, 1}), 0.5);
    check(close(batch.convex_hull(batch).c(), {0, 0, 4, 0}, 0), b + ": a batch");
}

/// Stated property: points of both capsules and of the segments between them lie in the hull.
void convex_hull_property(const std::string &b) {
    Rng rng(23);
    for (int k = 0; k < 8; ++k) {
        const Capsule C1 = test::random_capsule(rng, 3), C2 = test::random_capsule(rng, 3);
        const Capsule H = C1.convex_hull(C2);
        const Tensor p1 = test::sample_points(C1, rng, 20), p2 = test::sample_points(C2, rng, 20);
        check(H.contains(p1) && H.contains(p2), b + ": both capsules are inside");
        check(H.contains((p1 + p2) * 0.5), b + ": so are the midpoints of pairs");
        check(H.contains(C1) && H.contains(C2), b + ": as capsules");
    }
}

/// Capsules of different dimensions are refused; with the checks off a valid call gives the same.
void checks_and_their_switch() {
    const Capsule C1(column({0.0, 0.0}), column({1.0, 0.0}), 0.5);
    check(test::raises([&] { C1.convex_hull(Capsule(column({1.0}))); }, Error::dimension_mismatch),
          "convex_hull: the dimensions agree");
    const Capsule on = C1.convex_hull(C1);
    ChecksScope off(false);
    check(close(C1.convex_hull(C1).r(), on.r(), 0), "convex_hull: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        convex_hull_known(b);
        convex_hull_property(b);
    });
    checks_and_their_switch();
    return test::finish("capsule convex_hull");
}
