// test_empty_set_contains - EmptySet contains: only the empty sets
//
// CORA: contSet/@emptySet/contains_.m
// PROPERTY: contSet/@emptySet/contains_.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/registry.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void only_the_empty_set_is_contained(const std::string &b) {
    const EmptySet E(2);
    check(E.contains(EmptySet(2)), b + ": the empty set");
    check(E.contains(Interval::empty(2)), b + ": the empty interval");
    check(!E.contains(Interval(test::column({0.0, 0.0}), test::column({0.0, 0.0}))),
          b + ": a point is not contained");
    check(E.contains(Tensor::zeros({2, 0})), b + ": no points");
    check(!E.contains(test::column({0.0, 0.0})), b + ": a point");
    // random nonempty sets are never contained, also through the registry
    Rng rng(14);
    for (int trial = 0; trial < 10; ++trial) {
        const Zonotope Z = Zonotope::generate_random(rng, 2, std::nullopt, 3);
        check(!E.contains(Z) && !contains(E, Z), b + ": random zonotope");
        check(!E.contains(Interval::stack({Z.interval(), Z.interval()})), b + ": a batch of boxes");
    }
    check(contains(E, Interval::empty(2)), b + ": registry, empty interval");
}

/// A set of another dimension, a negative tol, and points of another dimension are refused.
void checks_and_their_switch() {
    const EmptySet E(2);
    check(test::raises([&] { E.contains(EmptySet(3)); }, Error::dimension_mismatch),
          "contains: a set of 3 dimensions");
    check(test::raises([&] { E.contains(EmptySet(2), -1.0); }, Error::wrong_value),
          "contains: tol = -1");
    check(
        test::raises([&] { E.contains(test::column({0.0, 0.0, 0.0})); }, Error::dimension_mismatch),
        "contains: a point of 3 dimensions");
    ChecksScope off(false);
    check(E.contains(EmptySet(2)), "contains: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(only_the_empty_set_is_contained);
    checks_and_their_switch();
    return test::finish("empty_set contains");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
