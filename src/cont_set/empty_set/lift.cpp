// lift - the empty set placed into a higher-dimensional space, as CORA's emptySet.lift_
//
// Syntax:   EmptySet R = E.lift(N, dims);
// Inputs:   N - dimension of the space, N >= dim;  dims - the N-dimensional coordinates (0-based)
//           of the dim coordinates of E
// Outputs:  R - the empty set of N dimensions
// See also: project, Interval::lift
//
// CORA: contSet/@emptySet/lift_.m

#include "cora/cont_set/empty_set/empty_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

EmptySet EmptySet::lift(int64_t N, const std::vector<int64_t> &dims) const {
    detail::Entry entry;
    CORA_CHECK(N >= n_, Error::wrong_value, [&] {
        return "lift: the space of " + std::to_string(N) +
               " dimensions must not be smaller than the set of " + std::to_string(n_);
    });
    CORA_CHECK(static_cast<int64_t>(dims.size()) == n_, Error::wrong_value, [&] {
        return "lift: one coordinate per dimension of the set is needed, got " +
               std::to_string(dims.size()) + " for " + std::to_string(n_);
    });
    for (const int64_t d : dims)
        CORA_CHECK(d >= 0 && d < N, Error::wrong_value, [&] {
            return "lift: the coordinate " + std::to_string(d) + " is not in [0, " +
                   std::to_string(N) + ")";
        });
    return EmptySet(N);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
