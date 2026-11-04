// center - the center of a capsule, as CORA's capsule.center
//
// Syntax:   c = C.center();
// Inputs:   -
// Outputs:  c - the stored center c (..., n, 1), the middle of the axis; for the empty capsule a
//           column without entries (n, 0), as CORA's, which marks an empty set. A batch gives
//           the stored centers
// See also: Capsule::c, ContSet::center
//
// CORA: contSet/@capsule/center.m

#include "cora/cont_set/capsule/capsule.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Capsule::center_impl() const {
    // a batch cannot hold members of different shapes
    if (c_.ndim() == 2 && is_empty()) return c_.zeros_like({dim(), 0});
    return c_;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
