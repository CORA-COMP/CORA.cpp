// reduce - the order reduction of an ellipsoid, as CORA's ellipsoid.reduce
//
// An ellipsoid has no generators to reduce, so CORA's reduce returns it unchanged.
//
// Syntax:   Eres = E.reduce(order);
// Inputs:   order - the target order, checked by the entry
// Outputs:  Eres - the same ellipsoid
// See also: Capsule::reduce
//
// CORA: contSet/@ellipsoid/reduce.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set Ellipsoid::reduce_impl(int) const { return *this; }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
