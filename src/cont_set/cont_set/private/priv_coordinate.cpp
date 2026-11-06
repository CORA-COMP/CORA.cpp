// priv_coordinate - what the entries that take a coordinate of the set refuse
//
// A coordinate outside [0, n).
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

#include "cont_set/cont_set/private/priv_coordinate.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

detail::Refusal priv_coordinate_refusal(int64_t N, int64_t n, const char *who) {
    if (N >= 0 && N < n) return std::nullopt;
    return std::pair{Error::wrong_value, std::string(who) + ": the coordinate " +
                                             std::to_string(N) + " is not in [0, " +
                                             std::to_string(n) + ")"};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
