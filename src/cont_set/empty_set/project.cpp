// project - the projection of the empty set, as CORA's emptySet.project
//
// Syntax:   Set P = E.project(dims);
// Inputs:   dims - the dimensions to keep, 0-based (checked by ContSet::project)
// Outputs:  P - the empty set with one dimension per entry of dims
// See also: lift, Interval::project
//
// CORA: contSet/@emptySet/project.m

#include "cora/cont_set/empty_set/empty_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set EmptySet::project_impl(const std::vector<int64_t> &dims) const {
    return EmptySet(static_cast<int64_t>(dims.size()));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
