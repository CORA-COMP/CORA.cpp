// test_zono_bundle_generate_random - ZonoBundle::generate_random: a random bundle
//
// CORA: contSet/@zonoBundle/generateRandom.m
// CORA authors: Mark Wetzlinger, Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "zono_bundle_testing.h"

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        Rng rng(71);
        const ZonoBundle zono_bundle = ZonoBundle::generate_random(rng, 3, 4);
        check(zono_bundle.dim() == 3 && zono_bundle.parallel_sets() == 4,
              b + ": the dimension and the number of members");
        check(!zono_bundle.representsa("emptySet"), b + ": the members share a point");
        const ZonoBundle any = ZonoBundle::generate_random(rng);
        check(any.dim() >= 1 && any.dim() <= 10 && any.parallel_sets() >= 2 &&
                  any.parallel_sets() <= 5,
              b + ": the defaults");
    });
    Rng rng(72);
    // wrong input is refused; the same call works with the checks off
    check(test::raises([&] { ZonoBundle::generate_random(rng, 0); }, Error::wrong_value),
          "generate_random: dimension 0");
    check(test::raises([&] { ZonoBundle::generate_random(rng, 2, 0); }, Error::wrong_value),
          "generate_random: no zonotopes");
    ChecksScope off(false);
    check(ZonoBundle::generate_random(rng, 2, 2).dim() == 2,
          "generate_random: the same with the checks off");
    return test::finish("zono_bundle generate_random");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
