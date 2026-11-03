// center - the center of a set, as CORA's contSet.center
//
// Syntax:   c = S.center();
// Inputs:   S - set
// Outputs:  c - the center, a column (..., n, 1)
// See also: interval, Zonotope::center_impl
//
// CORA: contSet/@contSet/center.m

#include "cora/cont_set/cont_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor ContSet::center() const {
    detail::Entry entry;
    return center_impl();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
