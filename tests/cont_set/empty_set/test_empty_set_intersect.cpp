// test_empty_set_intersect - EmptySet intersect: the empty set, whatever the other set is
//
// CORA: contSet/@emptySet/and_.m
// PROPERTY: contSet/@emptySet/and_.m

#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

void the_intersection_is_empty(const std::string &b) {
    const EmptySet E(2);
    const Interval I(test::column({0.0, 0.0}), test::column({1.0, 1.0}));
    check(E.intersect(I).dim() == 2, b + ": with a box");
    check(E.intersect(E).dim() == 2, b + ": with itself");
    check(E.intersect(Interval::empty(2)).dim() == 2, b + ": with the empty interval");
    // random sets: the result is the empty set of the same dimension, also through the registry
    Rng rng(11);
    for (int trial = 0; trial < 10; ++trial) {
        const Zonotope Z = Zonotope::generate_random(2, 4, rng);
        const Set R = intersect(E, Z);
        check(R.is<EmptySet>() && R.as<EmptySet>().dim() == 2, b + ": random zonotope");
        check(intersect(Z, E).is<EmptySet>(), b + ": the other order of the operands");
    }
}

/// A set of another dimension is refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const EmptySet E(2);
    const Interval I3(test::column({0.0, 0.0, 0.0}), test::column({1.0, 1.0, 1.0}));
    check(test::raises([&] { E.intersect(I3); }, Error::dimension_mismatch),
          "intersect: a box of 3 dimensions");
    ChecksScope off(false);
    check(E.intersect(I3).dim() == 2, "intersect: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(the_intersection_is_empty);
    checks_and_their_switch();
    return test::finish("empty_set intersect");
}
