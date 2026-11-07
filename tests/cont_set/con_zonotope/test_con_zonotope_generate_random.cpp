// test_con_zonotope_generate_random - ConZonotope::generate_random: a random constrained zonotope
//
// CORA: contSet/@conZonotope/generateRandom.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "con_zonotope_testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        Rng rng(79);
        const ConZonotope cz = ConZonotope::generate_random(rng, 3, std::nullopt, 6, 2);
        check(cz.dim() == 3 && cz.generators().shape() == Shape{3, 6} &&
                  cz.A().shape() == Shape{2, 6},
              b + ": the sizes");
        check(!cz.representsa("emptySet"), b + ": the constraints have a solution in the box");
        const ConZonotope plain = ConZonotope::generate_random(rng, 2, std::nullopt, 3, 0);
        check(plain.A().shape()[0] == 0 && plain.generators().shape() == Shape{2, 3},
              b + ": no constraints");
        const ConZonotope centered =
            ConZonotope::generate_random(rng, std::nullopt, Tensor::from_data({1.0, 2.0}, {2, 1}));
        check(centered.dim() == 2 && centered.generators().shape().back() == 4,
              b + ": the center gives n");
        const ConZonotope any = ConZonotope::generate_random(rng);
        check(any.dim() >= 1 && any.dim() <= 10, b + ": the defaults");
    });
    Rng rng(80);
    // wrong input is refused; the same call works with the checks off
    check(test::raises([&] { ConZonotope::generate_random(rng, 0); }, Error::wrong_value),
          "generate_random: dimension 0");
    check(test::raises([&] { ConZonotope::generate_random(rng, 2, std::nullopt, 3, -1); },
                       Error::wrong_value),
          "generate_random: negative constraints");
    ChecksScope off(false);
    check(ConZonotope::generate_random(rng, 2).dim() == 2,
          "generate_random: the same with the checks off");
    return test::finish("con_zonotope generate_random");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
