// test_empty_set_generate_random - EmptySet generate_random: the empty set of a given or random
// dimension
//
// CORA: contSet/@emptySet/generateRandom.m

#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/global/rng.h"
#include "testing.h"

#include <set>

using namespace cora;
using test::check;

namespace {

void the_dimension_is_given_or_drawn(const std::string &b) {
    Rng rng(5);
    check(EmptySet::generate_random(rng, 4).dim() == 4, b + ": a given dimension");
    check(EmptySet::generate_random(rng, 0).dim() == 0, b + ": dimension 0");
    std::set<int64_t> seen;
    for (int i = 0; i < 200; ++i) {
        const int64_t n = EmptySet::generate_random(rng).dim();
        check(n >= 1 && n <= 10, b + ": a drawn dimension is in 1..10");
        seen.insert(n);
    }
    check(seen.size() == 10, b + ": all of 1..10 occur");
    Rng first(9), second(9);
    bool repeatable = true;
    for (int i = 0; i < 20; ++i)
        repeatable = repeatable && EmptySet::generate_random(first).dim() ==
                                       EmptySet::generate_random(second).dim();
    check(repeatable, b + ": the same seed gives the same dimensions");
}

/// A negative dimension is refused; with the checks off the dimension is still the one given.
void checks_and_their_switch() {
    Rng rng(5);
    check(test::raises([&] { EmptySet::generate_random(rng, -1); }, Error::wrong_value), "n = -1");
    ChecksScope off(false);
    check(EmptySet::generate_random(rng, 3).dim() == 3,
          "generate_random: the same with checks off");
}

} // namespace

int main() {
    test::for_each_backend(the_dimension_is_given_or_drawn);
    checks_and_their_switch();
    return test::finish("empty_set generate_random");
}
