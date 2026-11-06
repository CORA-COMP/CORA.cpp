// project - the projection of the empty set, as CORA's emptySet.project
//
// Syntax:   Set P = E.project(dims);
// Inputs:   dims - the dimensions to keep, 0-based (checked by ContSet::project)
// Outputs:  P - the empty set with one dimension per entry of dims
// See also: lift, Interval::project
//
// CORA: contSet/@emptySet/project.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set EmptySet::project_impl(const std::vector<int64_t> &dims) const {
    return EmptySet(static_cast<int64_t>(dims.size()));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
