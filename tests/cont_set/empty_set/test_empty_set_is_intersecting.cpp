// test_empty_set_is_intersecting - EmptySet is_intersecting: never
//
// CORA: contSet/@emptySet/isIntersecting_.m
// PROPERTY: contSet/@emptySet/isIntersecting_.m

#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

void nothing_meets_the_empty_set(const std::string &b) {
    const EmptySet E(2);
    check(!E.is_intersecting(EmptySet(2)), b + ": itself");
    check(!E.is_intersecting(Interval::unbounded(2)), b + ": the whole plane");
    Rng rng(15);
    for (int trial = 0; trial < 10; ++trial) {
        const Zonotope Z = Zonotope::generate_random(rng, 2, std::nullopt, 3);
        check(!E.is_intersecting(Z) && !is_intersecting(E, Z) && !is_intersecting(Z, E),
              b + ": random zonotope");
        check(!E.is_intersecting(Z, 1e3), b + ": not even with a large tol");
    }
}

/// A set of another dimension and a negative tol are refused; with the checks off, the same answer.
void checks_and_their_switch() {
    const EmptySet E(2);
    check(test::raises([&] { E.is_intersecting(EmptySet(3)); }, Error::dimension_mismatch),
          "is_intersecting: a set of 3 dimensions");
    check(test::raises([&] { E.is_intersecting(EmptySet(2), -1.0); }, Error::wrong_value),
          "is_intersecting: tol = -1");
    ChecksScope off(false);
    check(!E.is_intersecting(EmptySet(2)), "is_intersecting: the same with the checks off");
}

} // namespace

int main() {
    test::for_each_backend(nothing_meets_the_empty_set);
    checks_and_their_switch();
    return test::finish("empty_set is_intersecting");
}
