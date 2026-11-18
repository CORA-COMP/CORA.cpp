// center - the center of a capsule, as CORA's capsule.center
//
// For the empty capsule a column without entries (n, 0), as CORA's, which marks an empty set. A
// batch gives the stored centers.
//
// Syntax:   c = C.center();
// Inputs:   -
// Outputs:  c - the stored center c (..., n, 1), the middle of the axis
// See also: Capsule::c, ContSet::center
//
// CORA: contSet/@capsule/center.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       06-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cora/cont_set/capsule/capsule.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Capsule::center_impl() const {
    // a batch cannot hold members of different shapes
    if (cc().ndim() == 2 && is_empty()) return cc().zeros_like({dim(), 0});
    return cc();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
