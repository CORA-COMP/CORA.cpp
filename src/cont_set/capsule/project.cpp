// project - the projection of a capsule onto some dimensions, as CORA's capsule.project
//
// The center and the generator are projected, the radius stays: the projection of a capsule is a
// capsule.
//
// Syntax:   Cres = C.project(dims);
// Inputs:   dims - the 0-based dimensions, in the order of the result
// Outputs:  Cres - the projected capsule
// See also: lift
//
// CORA: contSet/@capsule/project.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/capsule/private/priv.h"
#include "cora/cont_set/capsule/capsule.h"
#include "global/format.h"
#include "global/select.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set Capsule::project_impl(const std::vector<int64_t> &dims) const {
    // the rows of the identity that are kept, so it runs on every backend and has gradients
    const Tensor P = select_dims(c_, dims, dim());
    return Capsule(Unchecked{}, P.matmul(c_), P.matmul(g_), r_);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
