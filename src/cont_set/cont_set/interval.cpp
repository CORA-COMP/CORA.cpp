// interval - the interval hull of a set, as CORA's contSet.interval
//
// Syntax:   I = S.interval();
// Inputs:   S - set
// Outputs:  I - the smallest axis-aligned box that contains S
// See also: support_function, Zonotope::interval_impl
//
// CORA: contSet/@contSet/interval.m

#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/interval/interval.h"
#include "cora/cont_set/set.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Interval ContSet::interval() const {
    detail::Entry entry;
    return interval_impl();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
