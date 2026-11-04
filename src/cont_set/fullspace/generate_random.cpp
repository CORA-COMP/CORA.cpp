// generate_random - a random whole space, as CORA's fullspace.generateRandom
//
// Syntax:   F = Fullspace::generate_random(rng);   F = Fullspace::generate_random(rng, 3);
// Inputs:   rng - random numbers (its seed makes the result repeatable);  dimension >= 0
//           (CORA's name-value pair 'Dimension'); default: a random integer in 1..10, as randi(10)
// Outputs:  F - the whole space of n dimensions
// See also: EmptySet::generate_random, Interval::generate_random
//
// CORA: contSet/@fullspace/generateRandom.m

#include "cora/cont_set/fullspace/fullspace.h"

#include "cora/global/rng.h"

#include "cont_set/cont_set/private/priv_random_dimension.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Fullspace Fullspace::generate_random(Rng &rng, std::optional<int64_t> dimension) {
    detail::Entry entry;
    CORA_CHECK(!dimension || *dimension >= 0, Error::wrong_value, [&] {
        return "generate_random: the dimension must be nonnegative, got " +
               std::to_string(*dimension);
    });
    return Fullspace(dimension ? *dimension : priv_random_dimension(rng));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
