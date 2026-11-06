// test_empty_set_plus - EmptySet plus: the empty set absorbs a sum
//
// CORA: contSet/@emptySet/plus.m
// PROPERTY: contSet/@emptySet/plus.m
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

void the_sum_is_empty(const std::string &b) {
    const EmptySet E(2);
    check(E.plus(test::column({1.0, 2.0})).dim() == 2, b + ": a translation");
    check(E.plus(E).dim() == 2, b + ": with itself");
    Rng rng(12);
    for (int trial = 0; trial < 10; ++trial) {
        const Zonotope Z = Zonotope::generate_random(rng, 2, std::nullopt, 3);
        check(plus(E, Z).is<EmptySet>() && plus(Z, E).is<EmptySet>(), b + ": random zonotope");
        check((Set(E) + Set(Z)).as<EmptySet>().dim() == 2, b + ": the operator + of sets");
    }
}

/// A set or a vector of another dimension is refused; with the checks off the result is the same.
void checks_and_their_switch() {
    const EmptySet E(2);
    check(test::raises([&] { E.plus(test::column({1.0, 2.0, 3.0})); }, Error::dimension_mismatch),
          "plus: a vector of 3 dimensions");
    check(test::raises([&] { E.plus(EmptySet(3)); }, Error::dimension_mismatch),
          "plus: a set of 3 dimensions");
    ChecksScope off(false);
    check(E.plus(EmptySet(3)).dim() == 2, "plus: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(the_sum_is_empty);
    checks_and_their_switch();
    return test::finish("empty_set plus");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
