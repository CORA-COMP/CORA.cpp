// project - the projection of the whole space, as CORA's fullspace.project
//
// Syntax:   Set P = F.project(dims);
// Inputs:   dims - the dimensions to keep, 0-based (checked by ContSet::project)
// Outputs:  P - the whole space with one dimension per entry of dims. A repeated entry gives an
//           enclosure, as in CORA (the image is a diagonal). CORA refuses R^0 with
//           Error::not_supported; here every dims is out of range in R^0 (Error::wrong_value)
// See also: lift, EmptySet::project
//
// CORA: contSet/@fullspace/project.m

#include "cora/cont_set/fullspace/fullspace.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set Fullspace::project_impl(const std::vector<int64_t> &dims) const {
    return Fullspace(static_cast<int64_t>(dims.size()));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
