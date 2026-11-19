// project - the projection of a box onto some dimensions, as CORA's interval.project
//
// Syntax:   Ip = I.project(dims);
// Inputs:   dims - the dimensions to keep (0-based), in the order they should appear
// Outputs:  Ip - the box of the selected bounds; an interval matrix cannot be projected
// See also: vertices, Zonotope::project
//
// CORA: contSet/@interval/project.m
// CORA authors: Mark Wetzlinger
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"
#include "cora/global/error.h"


// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set Interval::project_impl(const std::vector<int64_t> &dims) const {
    // the coordinates of a box, the rows of an interval matrix
    const int axis = -elem_rank_;
    return with(inf_.take(dims, axis), sup_.take(dims, axis));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
