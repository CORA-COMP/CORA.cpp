// ball_radius - the radius of the ball around the center that holds a capsule (CORA's radius)
//
// The generator's length plus the radius; 0 for the empty capsule (CORA's is empty).
//
// Syntax:   r = C.ball_radius();
// Inputs:   -
// Outputs:  r - |g|_2 + r, (...)
// See also: volume, Interval::ball_radius
//
// CORA: contSet/@capsule/radius.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       06-November-2026
// Last update:   20-November-2026
// Reviewed:      ---

#include "cont_set/capsule/private/priv.h"
#include "cora/cont_set/capsule/capsule.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Capsule::ball_radius() const {
    detail::Entry entry;
    return priv_norm(gc()).add(rc()).squeeze(-1).squeeze(-1);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
