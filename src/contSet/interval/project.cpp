// project - the projection of a box onto some dimensions, as CORA's interval.project
//
// Syntax:   Ip = I.project(dims);
// Inputs:   dims - the dimensions to keep (0-based), in the order they should appear
// Outputs:  Ip - the box of the selected bounds; an interval matrix cannot be projected
// See also: vertices, Zonotope::project

#include "contSet/interval/interval.h"

#include "global/select.h"

#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::ct {


// ===========================================  MAIN  =========================================== //

std::unique_ptr<ContSet> Interval::project(const std::vector<int64_t> &dims) const {
    if (inf.shape().back() != 1)
        throw std::invalid_argument("Interval::project: only a box (column bounds) can be "
                                    "projected, not an interval matrix");
    const Tensor P = selectDims(inf, dims, dim());
    return std::make_unique<Interval>(P.matmul(inf), P.matmul(sup));
}

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
