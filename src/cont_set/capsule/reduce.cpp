// reduce - the order reduction of a capsule, as CORA's capsule.reduce
//
// A capsule has one generator, so CORA's reduce returns it unchanged.
//
// Syntax:   Cres = C.reduce(order);
// Inputs:   order - the target order, checked by the entry
// Outputs:  Cres - the same capsule
// See also: Zonotope::reduce
//
// CORA: contSet/@capsule/reduce.m
// CORA authors: Matthias Althoff
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

Set Capsule::reduce_impl(int) const { return *this; }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
