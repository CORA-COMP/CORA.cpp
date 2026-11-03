// minkowski_difference - the Minkowski difference of two sets, as CORA's contSet.minkDiff
//
// Syntax:   R = S.minkowski_difference(S2);   R = S.minkowski_difference(S2, type);
// Inputs:   S, S2 - sets of one dimension;  type - the algorithm of the class of S (Interval:
//           "inner", "exact"; Zonotope: "approx", "inner", "outer", ...); default: the class's own
// Outputs:  R - {x | x + S2 in S}, or an inner or outer approximation as the class defines
// See also: minus, Interval::minkowski_difference, Zonotope::mink_diff
//
// CORA: contSet/@contSet/minkDiff.m

#include "cont_set/cont_set/private/priv_noops.h"
#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set ContSet::minkowski_difference(const ContSet &S2, const std::optional<std::string> &type) const {
    detail::Entry entry;
    CORA_CHECK(S2.dim() == dim(), Error::dimension_mismatch, [&] {
        return "minkowski_difference: the sets differ in dimension, " + std::to_string(dim()) +
               " and " + std::to_string(S2.dim());
    });
    return minkowski_difference_impl(S2, type);
}

Set ContSet::minkowski_difference_impl(const ContSet &, const std::optional<std::string> &) const {
    priv_noops("minkowski_difference", *this);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
