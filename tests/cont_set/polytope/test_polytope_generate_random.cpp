// test_polytope_generate_random - Polytope::generate_random: random sets with CORA's options
//
// CORA: contSet/@polytope/generateRandom.m
// PROPERTY: contSet/@polytope/generateRandom.m
//
// Property: a polytope has the asked dimension, number of rows, boundedness and degeneracy
//
// CORA authors: Mark Wetzlinger, Adrian Kulmburg
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "polytope_testing.h"

using namespace cora;
using test::check;
using test::close;
using test::column;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The options are met.
void options(const std::string &b) {
    Rng rng(4);
    bool dimensions = true, rows = true, bounded = true, unbounded = true, flat = true, full = true;
    for (int n = 1; n <= 4; ++n) {
        const Polytope P = Polytope::generate_random(rng, n, 2 * n + 2);
        dimensions = dimensions && P.dim() == n;
        rows = rows && P.A().shape()[0] == 2 * n + 2;
        bounded = bounded && P.is_bounded() && P.is_full_dim();
        // the unbounded and the degenerate ones
        const Polytope open = Polytope::generate_random(rng, n, 2 * n + 2, false, false);
        unbounded = unbounded && !open.is_bounded();
        if (n > 1) {
            const Polytope deg = Polytope::generate_random(rng, n, 2 * n + 2, true, true);
            flat = flat && !deg.is_full_dim() && deg.dim() == n;
        }
        full = full && Polytope::generate_random(rng, n, std::nullopt, false).is_full_dim();
    }
    check(dimensions && rows, b + ": dimension and number of rows");
    check(bounded && unbounded && flat && full, b + ": bounded, unbounded, degenerate");
}

/// The defaults: a dimension of 1..10, rows 2n + n/2 .. 2n + n.
void defaults(const std::string &b) {
    Rng rng(4);
    bool ok = true;
    for (int k = 0; k < 10; ++k) {
        const Polytope P = Polytope::generate_random(rng);
        ok = ok && P.dim() >= 1 && P.dim() <= 10 && P.A().shape()[0] >= 2 * P.dim() + P.dim() / 2 &&
             P.A().shape()[0] <= 3 * P.dim() && P.is_bounded();
    }
    check(ok, b + ": the defaults");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        options(b);
        defaults(b);
    });
    Rng rng(4);
    // CORA's refusals of inconsistent options
    check(test::raises([&] { Polytope::generate_random(rng, 3, 3, std::nullopt, true); },
                       Error::wrong_value),
          "generate_random: bounded needs n + 1 rows");
    check(test::raises([&] { Polytope::generate_random(rng, 3, 1, true, std::nullopt); },
                       Error::wrong_value),
          "generate_random: one row is not degenerate");
    check(test::raises([&] { Polytope::generate_random(rng, 1, std::nullopt, true, false); },
                       Error::wrong_value),
          "generate_random: 1D degenerate and unbounded");
    check(test::raises([&] { Polytope::generate_random(rng, 0); }, Error::wrong_value),
          "generate_random: dimension >= 1");
    ChecksScope off(false);
    check(Polytope::generate_random(rng, 2).dim() == 2,
          "generate_random: the same with the checks off");
    return test::finish("polytope generate_random");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
