// minkowski_difference - the Minkowski difference of two time intervals
//
// Syntax:   K = I.minkowski_difference(sub);   K = I.minkowski_difference(sub, "inner");
// Inputs:   sub - an interval; type - "exact" or "inner"
// Outputs:  K - the Minkowski difference
// See also: stl_interval.h
//
// CORA: specification/@stlInterval/minkDiff.m
// CORA authors: Florian Lercher
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/specification/stl_interval.h"

#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

// Unbounded minus unbounded stays unbounded, where CORA gets NaN (docs/cora-differences.md).
StlInterval StlInterval::minkowski_difference(const StlInterval &sub,
                                              const std::string &type) const {
    detail::Entry entry;
    CORA_CHECK(type == "exact" || type == "inner", Error::wrong_value,
               [] { return std::string("minkowski_difference: type is 'exact' or 'inner'"); });
    if (sub.empty_) return {0, std::numeric_limits<double>::infinity(), true, false};
    if (empty_) return {};
    if (!std::isinf(upper_) && std::isinf(sub.upper_)) return {}; // finite - inf: nothing is left
    const double lb = lower_ - sub.lower_;
    const double ub = std::isinf(upper_) ? upper_ : upper_ - sub.upper_;
    const bool lc = lc_ || !sub.lc_, rc = rc_ || !sub.rc_;
    if (lb < 0) return {0, ub, true, rc};
    return {lb, ub, lc, rc};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
