// priv_coordinate - what the entries that take a coordinate of the set refuse
//
// Syntax:   detail::Refusal r = priv_coordinate_refusal(N, n, "split");
// Inputs:   N - the coordinate (0-based)
//           n - the dimension of the set
//           who - the name of the operation, for the message
// Outputs:  r - the identifier and the message of the refusal, or nothing
// See also: ConZonotope::split, ZonoBundle::split
//
// CORA.cpp only: CORA checks the coordinate in each split
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/cont_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

detail::Refusal priv_coordinate_refusal(int64_t N, int64_t n, const char *who);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
