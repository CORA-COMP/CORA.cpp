// supremum - the upper bound of a box, as CORA's interval.supremum
//
// Syntax:   Tensor u = I.supremum();
// Inputs:   -
// Outputs:  u - sup, of the shape of the bounds
// See also: Interval::sup, infimum
//
// CORA: contSet/@interval/supremum.m

#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Interval::supremum() const {
    detail::Entry entry;
    return sup_;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
