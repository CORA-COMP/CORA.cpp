// center - the center of a capsule, as CORA's capsule.center
//
// For the empty capsule NaN, as for the empty box (CORA's has no entries, (n, 0)). A batch gives
// the stored centers.
//
// Syntax:   c = C.center();
// Inputs:   -
// Outputs:  c - the stored center c (..., n), the middle of the axis
// See also: Capsule::c, ContSet::center
//
// CORA: contSet/@capsule/center.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       06-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cora/cont_set/capsule/capsule.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Capsule::center_impl() const {
    // a batch cannot hold members of different shapes
    if (c_.ndim() == 1 && is_empty())
        return c_.full_like(c_.shape(), std::numeric_limits<double>::quiet_NaN());
    return c_;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
