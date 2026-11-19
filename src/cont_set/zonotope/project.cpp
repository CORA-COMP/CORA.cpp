// project - the projection of a zonotope onto some dimensions, as CORA's zonotope.project
//
// Syntax:   Zp = Z.project(dims);
// Inputs:   dims - the dimensions to keep (0-based), in the order they should appear
// Outputs:  Zp - the zonotope {P x : x in Z}, exact: the selected rows of c and G
// See also: vertices, Interval::project
//
// CORA: contSet/@zonotope/project.m
// CORA authors: Matthias Althoff
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"

#include "global/select.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set Zonotope::project_impl(const std::vector<int64_t> &dims) const {
    const Tensor P = select_dims(c_, dims, dim());
    auto [c, G] = Tensor::zonotope_linear_map(P, c_, G_);
    return Zonotope(std::move(c), std::move(G));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
