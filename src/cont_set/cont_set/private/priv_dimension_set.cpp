// priv_dimension_set - why a lift of a set given by its dimension alone is refused
//
// Syntax:   std::string why = E.lift_refusal(N, dims, "lift");
// Inputs:   N - the dimension of the larger space
//           dims - the position (0-based) of each of the
//           n dimensions of the set in it
// Outputs:  why - "" if lift(N, dims) is valid, else the message of its Error::wrong_value
// See also: cont_set/dimension_set.h, EmptySet::lift, Fullspace::lift
//
// CORA.cpp only: CORA's emptySet and fullspace each check their lift themselves.
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/dimension_set.h"

#include <string>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::detail {


// ===========================================  MAIN  =========================================== //

std::string DimensionSet::lift_refusal(int64_t N, const std::vector<int64_t> &dims,
                                       const char *operation) const {
    const std::string name = std::string(operation) + ": ";
    if (N < n_)
        return name + "the space of " + std::to_string(N) +
               " dimensions must not be smaller than the set of " + std::to_string(n_);
    if (static_cast<int64_t>(dims.size()) != n_)
        return name + "one position per dimension of the set is needed, got " +
               std::to_string(dims.size()) + " for " + std::to_string(n_);
    for (const int64_t d : dims)
        if (d < 0 || d >= N)
            return name + "the position " + std::to_string(d) + " is not in [0, " +
                   std::to_string(N) + ")";
    return "";
}

} // namespace cora::detail

// ---------------------------------------  END OF CODE  ---------------------------------------- //
