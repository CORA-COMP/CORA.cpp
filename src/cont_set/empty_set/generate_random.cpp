// generate_random - a random empty set, as CORA's emptySet.generateRandom
//
// Syntax:   E = EmptySet::generate_random(rng);   E = EmptySet::generate_random(rng, 3);
// Inputs:   rng - random numbers (its seed makes the result repeatable);  n - dimension, n >= 0
//           (CORA's name-value pair 'Dimension'); default: a random integer in 1..10, as randi(10)
// Outputs:  E - the empty set of n dimensions
// See also: Fullspace::generate_random, Interval::generate_random
//
// CORA: contSet/@emptySet/generateRandom.m

#include "cora/cont_set/empty_set/empty_set.h"

#include "cora/global/rng.h"

#include "cont_set/cont_set/private/priv_random_dimension.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

EmptySet EmptySet::generate_random(Rng &rng, std::optional<int64_t> n) {
    detail::Entry entry;
    CORA_CHECK(!n || *n >= 0, Error::wrong_value, [&] {
        return "generate_random: the dimension must be nonnegative, got " + std::to_string(*n);
    });
    return EmptySet(n ? *n : priv_random_dimension(rng));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
