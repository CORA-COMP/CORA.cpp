// lift - the empty set placed into a higher-dimensional space, as CORA's emptySet.lift_
//
// Syntax:   EmptySet R = E.lift(N, dims);
// Inputs:   N - dimension of the space, N >= dim
//           dims - the N-dimensional coordinates (0-based) of the dim coordinates of E
// Outputs:  R - the empty set of N dimensions
// See also: project, Interval::lift
//
// CORA: contSet/@emptySet/lift_.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

EmptySet EmptySet::lift(int64_t N, const std::vector<int64_t> &dims) const {
    detail::Entry entry;
    CORA_CHECK(lift_refusal(N, dims).empty(), Error::wrong_value,
               [&] { return lift_refusal(N, dims); });
    return EmptySet(N);
}

Set EmptySet::lift_impl(int64_t N, const std::vector<int64_t> &proj) const { return lift(N, proj); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
