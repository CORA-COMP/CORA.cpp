// and_ - the intersection of two time intervals
//
// Syntax:   StlInterval K = I.and_(J);
// Inputs:   other - the other interval
// Outputs:  K - the intersection
// See also: stl_interval.h
//
// CORA: specification/@stlInterval/and_.m
// CORA authors: Florian Lercher
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/specification/stl_interval.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

StlInterval StlInterval::and_(const StlInterval &other) const {
    detail::Entry entry;
    if (empty_ || other.empty_) return {};
    const double lb = std::max(lower_, other.lower_);
    const double ub = std::min(upper_, other.upper_);
    // the end of the intersection is closed where the end that decides is, or both when they tie
    const bool lc = other.lower_ < lower_   ? lc_
                    : lower_ < other.lower_ ? other.lc_
                                            : (lc_ && other.lc_);
    const bool rc = upper_ < other.upper_   ? rc_
                    : other.upper_ < upper_ ? other.rc_
                                            : (rc_ && other.rc_);
    return {lb, ub, lc, rc};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
