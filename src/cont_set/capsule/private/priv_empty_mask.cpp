// priv_empty_mask - which capsules of a batch are empty
//
// Syntax:   Tensor empty = priv_empty_mask(r);
// Inputs:   r - the radii (..., 1, 1)
// Outputs:  empty - 1 where the radius is negative (the empty capsule has the radius -1), else 0
// See also: Capsule::empty, Capsule::is_empty
//
// CORA.cpp only: MATLAB marks the empty capsule by an empty radius; here it is the radius -1
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/capsule/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor priv_empty_mask(const Tensor &r) { return r.less(r.zeros_like()); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
