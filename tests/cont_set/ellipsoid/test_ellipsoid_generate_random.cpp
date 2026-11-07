// test_ellipsoid_generate_random - Ellipsoid generate_random: given or drawn dimension, center
//
// CORA: contSet/@ellipsoid/generateRandom.m
// PROPERTY: contSet/@ellipsoid/generateRandom.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/rng.h"
#include "ellipsoid_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The given dimension and center are used; Q is symmetric, positive semidefinite, and below 10.
void generate_random_property(const std::string &b) {
    Rng rng(151);
    const Ellipsoid given = Ellipsoid::generate_random(rng, 4, false, column({1.0, 2.0, 3.0, 4.0}));
    check(given.dim() == 4 && close(given.q(), {1, 2, 3, 4}, 0), b + ": given");
    check(Ellipsoid::generate_random(rng, std::nullopt, false, column({1.0, 2.0})).dim() == 2,
          b + ": the dimension of the center");
    for (int k = 0; k < 8; ++k) {
        const Ellipsoid E = Ellipsoid::generate_random(rng, 5);
        check(E.Q().abs().max().item() <= 10.0 + 1e-9 && E.is_full_dim(),
              b + ": entries below 10, full rank");
        const Ellipsoid D = Ellipsoid::generate_random(rng, 5, true);
        check(!D.is_full_dim() && D.Q().eigh().first.min().item() > -1e-9,
              b + ": degenerate and semidefinite");
        const Ellipsoid any = Ellipsoid::generate_random(rng);
        check(any.dim() >= 1 && any.dim() <= 30, b + ": dimension in 1..30");
    }
}

/// A bad dimension and a center that disagrees are refused; with the checks off a valid call works.
void checks_and_their_switch() {
    Rng rng(1);
    check(test::raises([&] { Ellipsoid::generate_random(rng, 0); }, Error::wrong_value),
          "generate_random: n");
    check(test::raises([&] { Ellipsoid::generate_random(rng, 3, false, column({1.0})); },
                       Error::dimension_mismatch),
          "generate_random: center");
    ChecksScope off(false);
    check(Ellipsoid::generate_random(rng, 2).dim() == 2,
          "generate_random: works with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { generate_random_property(b); });
    checks_and_their_switch();
    return test::finish("ellipsoid generate_random");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
