// project - the projection of a constrained zonotope onto some dimensions, as CORA's project
//
// Syntax:   S = cz.project(dims);
// Inputs:   dims - the dimensions to keep (0-based), in the order they appear; repeats allowed
// Outputs:  S - the constrained zonotope of the selected rows, the constraints unchanged: exact
// See also: lift, Zonotope::project_impl
//
// CORA: contSet/@conZonotope/project.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set ConZonotope::project_impl(const std::vector<int64_t> &dims) const {
    return ConZonotope(c_.expand_dims(-1).take(dims, -2).squeeze(-1),
                       (G_.matrix_transpose().take(dims, -2)).matrix_transpose(), F_);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
