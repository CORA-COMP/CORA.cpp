// project_high_dim - the empty set placed into a larger space, as CORA's emptySet.projectHighDim_
//
// Syntax:   EmptySet R = E.project_high_dim(N, proj);
// Inputs:   N - dimension of the space, N >= dim;  proj - the N-dimensional coordinates (0-based)
//           of the dim coordinates of E
// Outputs:  R - the empty set of N dimensions
// See also: lift, Interval::project_high_dim
//
// CORA: contSet/@emptySet/projectHighDim_.m
// CORA: contSet/@contSet/projectHighDim.m

#include "cora/cont_set/empty_set/empty_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

EmptySet EmptySet::project_high_dim(int64_t N, const std::vector<int64_t> &proj) const {
    detail::Entry entry;
    CORA_CHECK(lift_refusal(N, proj, "project_high_dim").empty(), Error::wrong_value,
               [&] { return lift_refusal(N, proj, "project_high_dim"); });
    return EmptySet(N);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
