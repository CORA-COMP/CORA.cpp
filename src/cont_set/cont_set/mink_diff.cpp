// mink_diff - the Minkowski difference of two sets, as CORA's contSet.minkDiff
//
// Interval takes "inner" and "exact"; Zonotope "approx", "inner", "outer", ...
//
// Syntax:   R = S.mink_diff(S2);   R = S.mink_diff(S2, method);
// Inputs:   S, S2 - sets of one dimension
//           method - the algorithm of the class of S; default: the class's own
// Outputs:  R - {x | x + S2 in S}, or an inner or outer approximation as the class defines
// See also: minus, Interval::mink_diff, Zonotope::mink_diff
//
// CORA: contSet/@contSet/minkDiff.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/cont_set/private/priv_noops.h"
#include "cora/cont_set/cont_set.h"
#include "cora/cont_set/set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set ContSet::mink_diff(const ContSet &S2, const std::optional<std::string> &method) const {
    detail::Entry entry;
    CORA_CHECK(S2.dim() == dim(), Error::dimension_mismatch, [&] {
        return "mink_diff: the sets differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(S2.dim());
    });
    entry.release();
    return mink_diff_impl(S2, method);
}

Set ContSet::mink_diff_impl(const ContSet &, const std::optional<std::string> &) const {
    priv_noops("mink_diff", *this);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
