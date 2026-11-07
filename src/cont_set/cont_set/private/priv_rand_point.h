// priv_rand_point - what the entries of rand_point with a type refuse
//
// Syntax:   detail::Refusal r = priv_rand_point_refusal(N, type);
// Inputs:   N - the number of points
//           type - "standard" or "extreme"
// Outputs:  r - the identifier and the message of the refusal, or nothing
// See also: Zonotope::rand_point, ConZonotope::rand_point, ZonoBundle::rand_point
//
// CORA.cpp only: CORA checks the type in each randPoint_
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/cont_set.h"

#include <string>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

detail::Refusal priv_rand_point_refusal(int64_t N, const std::string &type);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
