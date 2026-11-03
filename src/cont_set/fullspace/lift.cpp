// lift - the whole space in a space of higher dimension, as CORA's fullspace.lift_
//
// Syntax:   Fullspace R = F.lift(N, dims);
// Inputs:   N - the dimension of the new space, N >= n;  dims - the position (0-based) of each
//           dimension of F in it; CORA does not check it, CORA.cpp as Interval::lift
// Outputs:  R - R^N: the new dimensions are unbounded, as the old ones already are
// See also: Interval::lift
//
// CORA: contSet/@fullspace/lift_.m

#include "cora/cont_set/fullspace/fullspace.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Fullspace Fullspace::lift(int64_t N, const std::vector<int64_t> &dims) const {
    detail::Entry entry;
    CORA_CHECK(N >= n_, Error::wrong_value, [&] {
        return "lift: the space of " + std::to_string(N) +
               " dimensions must not be smaller than the set of " + std::to_string(n_);
    });
    CORA_CHECK(static_cast<int64_t>(dims.size()) == n_, Error::wrong_value, [&] {
        return "lift: one position per dimension is needed, got " + std::to_string(dims.size()) +
               " for " + std::to_string(n_);
    });
    for (const int64_t d : dims)
        CORA_CHECK(d >= 0 && d < N, Error::wrong_value, [&] {
            return "lift: the position " + std::to_string(d) + " is not in [0, " +
                   std::to_string(N) + ")";
        });
    return Fullspace(N);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
