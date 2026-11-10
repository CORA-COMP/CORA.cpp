// priv_embed - a box placed into the coordinates of a higher-dimensional space
//
// See also: priv_embed.h
//
// CORA.cpp only: a helper of project and lift.
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/interval/private/priv_embed.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval priv_embed(const Interval &I, int64_t N, const std::vector<int64_t> &dims, double lower,
                    double upper) {
    // row j of the bounds goes to coordinate dims[j]; the extra row m is the unoccupied one
    const int64_t m = I.dim();
    std::vector<int64_t> source(N, m);
    for (int64_t j = 0; j < m; ++j) source[dims[j]] = j;
    std::vector<int64_t> shape = I.inf().shape();
    shape[shape.size() - 2] = 1;
    const Tensor low = Tensor::concat({I.inf(), I.inf().full_like(shape, lower)}, -2);
    const Tensor high = Tensor::concat({I.sup(), I.sup().full_like(shape, upper)}, -2);
    return {low.take(source, -2), high.take(source, -2)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
