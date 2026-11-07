// radius - the radius of the ball around the center that holds a capsule, as CORA's capsule.radius
//
// The generator's length plus the radius; the empty capsule keeps its marker -1 radius.
//
// Syntax:   r = C.radius();
// Inputs:   -
// Outputs:  r - |g|_2 + r, (..., 1, 1)
// See also: volume, Interval::radius
//
// CORA: contSet/@capsule/radius.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/capsule/private/priv.h"
#include "cora/cont_set/capsule/capsule.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Capsule::radius() const {
    detail::Entry entry;
    return priv_norm(g_).add(r_);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
