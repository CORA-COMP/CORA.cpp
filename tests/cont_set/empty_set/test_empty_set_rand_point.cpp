// test_empty_set_rand_point - EmptySet rand_point: no points, whatever N is
//
// CORA: contSet/@emptySet/randPoint_.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"
#include "testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void there_is_no_point_to_draw(const std::string &b) {
    Rng rng(3);
    const EmptySet E(2);
    check(E.rand_point(5, rng).shape() == std::vector<int64_t>({2, 0}), b + ": 5 requested, none");
    check(E.rand_point(0, rng).shape() == std::vector<int64_t>({2, 0}), b + ": 0 requested");
}

/// A negative count is refused; with the checks off the result is the same.
void checks_and_their_switch() {
    Rng rng(3);
    const EmptySet E(2);
    check(test::raises([&] { E.rand_point(-1, rng); }, Error::wrong_value), "rand_point: N = -1");
    ChecksScope off(false);
    check(E.rand_point(3, rng).shape() == std::vector<int64_t>({2, 0}),
          "rand_point: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend(there_is_no_point_to_draw);
    checks_and_their_switch();
    return test::finish("empty_set rand_point");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
