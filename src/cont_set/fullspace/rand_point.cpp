// rand_point - random points of the whole space, as CORA's fullspace.randPoint_
//
// Syntax:   P = F.rand_point(N, rng);
// Inputs:   N - number of points
//           rng - random numbers
// Outputs:  P - N standard normal points as columns (n, N), CORA's type 'standard'. R^0 is
//           Error::not_supported, as in CORA
// See also: center, Interval::rand_point
//
// CORA: contSet/@fullspace/randPoint_.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/fullspace/fullspace.h"

#include "cora/global/rng.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

CORA_FROM_NUMBERS Tensor Fullspace::rand_point_impl(int64_t N, Rng &rng) const {
    if (n_ == 0) CORA_FAIL(Error::not_supported, "rand_point: sampling of R^0 is not supported");
    std::vector<double> points(n_ * N);
    rng.normal(points.data(), points.size(), 1.0);
    return Tensor::from_data(points, {n_, N});
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
