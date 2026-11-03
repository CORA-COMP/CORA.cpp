// test_empty_set_convex_hull - EmptySet convex_hull: the other set itself
//
// CORA: contSet/@emptySet/convHull_.m
// PROPERTY: contSet/@emptySet/convHull_.m

#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

void the_hull_is_the_other_set(const std::string &b) {
    const EmptySet E(2);
    const Interval I(test::column({0.0, -1.0}), test::column({1.0, 3.0}));
    const Set R = E.convex_hull(Set(I));
    check(R.is<Interval>() && R.as<Interval>().isequal(I, 0), b + ": a box stays a box");
    check(E.convex_hull(Set(E)).as<EmptySet>().dim() == 2, b + ": with itself");
    // random zonotopes come back with the same center and generators, through the registry too
    Rng rng(13);
    for (int trial = 0; trial < 10; ++trial) {
        const Zonotope Z = Zonotope::generate_random(2, 4, rng);
        for (const Set &H : {E.convex_hull(Set(Z)), convex_hull(E, Z), convex_hull(Z, E)}) {
            check(H.is<Zonotope>() && test::close(H.as<Zonotope>().center(), Z.center(), 0) &&
                      test::close(H.as<Zonotope>().generators(), Z.generators(), 0),
                  b + ": random zonotope");
        }
    }
}

/// A set of another dimension is refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const EmptySet E(2);
    check(test::raises([&] { E.convex_hull(Set(EmptySet(3))); }, Error::dimension_mismatch),
          "convex_hull: a set of 3 dimensions");
    ChecksScope off(false);
    check(E.convex_hull(Set(EmptySet(3))).as<EmptySet>().dim() == 3,
          "convex_hull: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(the_hull_is_the_other_set);
    checks_and_their_switch();
    return test::finish("empty_set convex_hull");
}
