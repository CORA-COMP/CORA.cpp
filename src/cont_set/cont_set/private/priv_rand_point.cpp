// priv_rand_point - what the entries of rand_point with a type refuse
//
// A negative number of points, and a type other than "standard" and "extreme".
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

#include "cont_set/cont_set/private/priv_rand_point.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

detail::Refusal priv_rand_point_refusal(int64_t N, const std::string &type) {
    if (N < 0)
        return std::pair{Error::wrong_value, "rand_point: the number of points must be "
                                             "nonnegative, got " + std::to_string(N)};
    if (type != "standard" && type != "extreme")
        return std::pair{Error::wrong_value, "rand_point: unknown type '" + type +
                                                 "'; use \"standard\" or \"extreme\""};
    return std::nullopt;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
