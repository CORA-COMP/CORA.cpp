// generate_random - a random set of an admissible class, as CORA's contSet.generateRandom
//
// Syntax:   Set S = ContSet::generate_random(rng);
//           Set S = ContSet::generate_random(rng, 3, {"interval"});
// Inputs:   rng - random numbers (seedable);  n - the dimension, default uniform in 1..10;
//           admissible - CORA names of the classes to draw from ("interval", "zonotope"); one is
//           drawn uniformly
// Outputs:  S - Interval::generate_random(n, rng) or Zonotope::generate_random(n, 2 n, rng)
// See also: Interval::generate_random, Zonotope::generate_random
//
// CORA: contSet/@contSet/generateRandom.m

#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/rng.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A uniform integer in 0..count-1.
int64_t aux_uniform_index(Rng &rng, int64_t count) {
    double u = 0.0;
    rng.uniform(&u, 1, 0.0, 1.0);
    return std::min<int64_t>(count - 1, static_cast<int64_t>(u * static_cast<double>(count)));
}

} // namespace


// ===========================================  MAIN  =========================================== //

CORA_FROM_NUMBERS Set ContSet::generate_random(Rng &rng, std::optional<int64_t> n,
                                               const std::vector<std::string> &admissible) {
    detail::Entry entry;
    CORA_CHECK(!admissible.empty(), Error::wrong_value, [] {
        return std::string("generate_random: name at least one admissible class");
    });
    CORA_CHECK(!n || *n >= 1, Error::wrong_value, [&] {
        return "generate_random: the dimension must be at least 1, got " + std::to_string(*n);
    });
    for (const std::string &name : admissible)
        CORA_CHECK(name == "interval" || name == "zonotope", Error::not_supported, [&] {
            return "generate_random: the class '" + name +
                   "' is not implemented; admissible are 'interval' and 'zonotope'";
        });
    const std::string &name = admissible[static_cast<std::size_t>(
        aux_uniform_index(rng, static_cast<int64_t>(admissible.size())))];
    const int64_t dimension = n ? *n : 1 + aux_uniform_index(rng, 10);
    if (name == "interval") return Interval::generate_random(dimension, rng);
    return Zonotope::generate_random(dimension, 2 * dimension, rng);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
