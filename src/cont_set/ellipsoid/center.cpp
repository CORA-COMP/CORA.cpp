// center - the center of an ellipsoid, as CORA's ellipsoid.center
//
// Syntax:   Tensor c = E.center();
// Inputs:   -
// Outputs:  c - the stored center q (..., n, 1), a point of the ellipsoid; for the empty
//           ellipsoid the column without entries (n, 0)
// See also: interval, ContSet::center
//
// CORA: contSet/@ellipsoid/center.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/ellipsoid/ellipsoid.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Ellipsoid::center_impl() const { return q_; }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
