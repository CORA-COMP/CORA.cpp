// project - the projection of a box onto some dimensions, as CORA's interval.project
//
// Syntax:   Ip = I.project(dims);
// Inputs:   dims - the dimensions to keep (0-based), in the order they should appear
// Outputs:  Ip - the box of the selected bounds; an interval matrix cannot be projected
// See also: vertices, Zonotope::project
//
// CORA: contSet/@interval/project.m

#include "cora/cont_set/interval/interval.h"
#include "cora/global/error.h"

#include "global/select.h"


// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set Interval::project_impl(const std::vector<int64_t> &dims) const {
    const Tensor P = select_dims(inf_, dims, dim());
    return Interval(P.matmul(inf_), P.matmul(sup_));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
