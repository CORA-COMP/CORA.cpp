// project - the projection of a zonotope onto some dimensions, as CORA's zonotope.project
//
// Syntax:   Zp = Z.project(dims);
// Inputs:   dims - the dimensions to keep (0-based), in the order they should appear
// Outputs:  Zp - the zonotope {P x : x in Z}, exact: the selected rows of c and G
// See also: vertices, Interval::project
//
// CORA: contSet/@zonotope/project.m

#include "cora/cont_set/zonotope/zonotope.h"

#include "global/select.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set Zonotope::project_impl(const std::vector<int64_t> &dims) const {
    const Tensor P = select_dims(c_, dims, dim());
    return Zonotope(P.matmul(c_), P.matmul(G_));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
