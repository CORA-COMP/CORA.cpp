// priv_random_dimension - the dimension CORA's generateRandom draws when none is given
//
// See also: priv_random_dimension.h
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/cont_set/private/priv_random_dimension.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

int64_t priv_random_dimension(Rng &rng) {
    double u = 0.0;
    rng.uniform(&u, 1, 0.0, 1.0);
    return std::min<int64_t>(10, 1 + static_cast<int64_t>(std::floor(10.0 * u)));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
