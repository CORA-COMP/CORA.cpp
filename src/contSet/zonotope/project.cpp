// project - the projection of a zonotope onto some dimensions, as CORA's zonotope.project
//
// Syntax:   Zp = Z.project(dims);
// Inputs:   dims - the dimensions to keep (0-based), in the order they should appear
// Outputs:  Zp - the zonotope {P x : x in Z}, exact: the selected rows of c and G
// See also: vertices, Interval::project

#include "contSet/zonotope/zonotope.h"

#include "global/select.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::ct {


// ===========================================  MAIN  =========================================== //

std::unique_ptr<ContSet> Zonotope::project(const std::vector<int64_t> &dims) const {
    const Tensor P = selectDims(c, dims, dim());
    return std::make_unique<Zonotope>(P.matmul(c), P.matmul(G));
}

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
