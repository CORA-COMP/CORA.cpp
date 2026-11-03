// priv_random_dimension - the dimension CORA's generateRandom draws when none is given
//
// Syntax:   int64_t n = priv_random_dimension(rng);
// Inputs:   rng - random numbers
// Outputs:  n - a uniform integer in 1..10, as MATLAB's randi(10)
// See also: EmptySet::generate_random, Fullspace::generate_random
//
// CORA.cpp only: CORA's generateRandom calls randi inside each class

#pragma once

#include "cora/global/rng.h"

#include <cstdint>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

int64_t priv_random_dimension(Rng &rng);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
