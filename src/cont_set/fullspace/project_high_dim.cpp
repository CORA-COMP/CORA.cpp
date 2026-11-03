// project_high_dim - the whole space in a larger space, as CORA's fullspace.projectHighDim_
//
// Syntax:   Fullspace R = F.project_high_dim(N, proj);
// Inputs:   N - dimension of the space, N >= dim;  proj - the N-dimensional coordinates (0-based)
//           of the dim coordinates of F
// Outputs:  R - the set itself, for N = dim. Any larger N would put the new coordinates at 0, which
//           the whole space cannot do: CORA refuses it with Error::not_defined (CORALean's
//           definition is R^N there, its row claims N = dim only)
// See also: lift, EmptySet::project_high_dim
//
// CORA: contSet/@fullspace/projectHighDim_.m
// CORA: contSet/@contSet/projectHighDim.m

#include "cora/cont_set/fullspace/fullspace.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Fullspace Fullspace::project_high_dim(int64_t N, const std::vector<int64_t> &proj) const {
    detail::Entry entry;
    CORA_CHECK(lift_refusal(N, proj, "project_high_dim").empty(), Error::wrong_value,
               [&] { return lift_refusal(N, proj, "project_high_dim"); });
    CORA_CHECK(N == n_, Error::not_defined, [] {
        return std::string("project_high_dim: the new dimensions cannot be bounded at 0 for the "
                           "whole space (see lift)");
    });
    return *this;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
