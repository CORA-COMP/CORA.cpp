// rand_point - random points of a set, checked, as CORA's contSet.randPoint
//
// Syntax:   x = S.rand_point(N, rng);
// Inputs:   N - number of points, nonnegative; rng - the random numbers
// Outputs:  x - N points of S as columns (..., n, N)
// See also: Zonotope::rand_point, Interval::rand_point_impl
//
// CORA: contSet/@contSet/randPoint.m

#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor ContSet::rand_point(int64_t N, Rng &rng) const {
    detail::Entry entry;
    CORA_CHECK(N >= 0, Error::wrong_value, [&] {
        return "rand_point: the number of points must be nonnegative, got " + std::to_string(N);
    });
    return rand_point_impl(N, rng);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
