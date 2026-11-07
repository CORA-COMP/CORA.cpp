// test_capsule_generate_random - Capsule generate_random: given or drawn dimension, center, radius
//
// CORA: contSet/@capsule/generateRandom.m
// PROPERTY: contSet/@capsule/generateRandom.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "capsule_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The given dimension, center and radius are used; the drawn ones stay in CORA's ranges.
void generate_random_property(const std::string &b) {
    Rng rng(83);
    const Capsule given = Capsule::generate_random(rng, 4, column({1.0, 2.0, 3.0, 4.0}), 0.25);
    check(given.dim() == 4 && close(given.c(), {1, 2, 3, 4}, 0) && close(given.r(), {0.25}, 0),
          b + ": given");
    check(Capsule::generate_random(rng, std::nullopt, column({1.0, 2.0})).dim() == 2,
          b + ": the dimension of the center");
    for (int k = 0; k < 10; ++k) {
        const Capsule C = Capsule::generate_random(rng);
        const double r = C.r().item();
        check(C.dim() >= 1 && C.dim() <= 10 && r >= 0 && r <= 5,
              b + ": dimension in 1..10 and r in [0, 5]");
        for (const double x : C.generator().data()) check(x >= 0 && x <= 1, b + ": g in [0, 1]");
    }
}

/// A bad dimension, a center that disagrees and a negative radius are refused.
void checks_and_their_switch() {
    Rng rng(1);
    check(test::raises([&] { Capsule::generate_random(rng, 0); }, Error::wrong_value),
          "generate_random: n");
    check(test::raises([&] { Capsule::generate_random(rng, 3, column({1.0})); },
                       Error::dimension_mismatch),
          "generate_random: center");
    check(test::raises([&] { Capsule::generate_random(rng, 2, std::nullopt, -1.0); },
                       Error::wrong_value),
          "generate_random: r");
    ChecksScope off(false);
    check(Capsule::generate_random(rng, 2).dim() == 2,
          "generate_random: works with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { generate_random_property(b); });
    checks_and_their_switch();
    return test::finish("capsule generate_random");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
