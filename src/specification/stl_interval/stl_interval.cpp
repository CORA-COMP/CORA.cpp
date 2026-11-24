// stl_interval - the constructor of a time interval, as CORA's stlInterval(lb, ub, lc, rc)
//
// Syntax:   StlInterval I(lb, ub, lc, rc);
// See also: stl_interval.h
//
// CORA: specification/@stlInterval/stlInterval.m
// CORA authors: Florian Lercher
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/specification/stl_interval.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

StlInterval::StlInterval(double lb, double ub, bool lc, bool rc) {
    detail::Entry entry;
    CORA_CHECK(!std::isnan(lb) && !std::isnan(ub), Error::wrong_input_in_constructor,
               [] { return std::string("StlInterval: the bounds must not be NaN"); });
    CORA_CHECK(lb >= 0, Error::wrong_input_in_constructor,
               [] { return std::string("StlInterval: the lower bound must be non-negative"); });
    // an infinite bound is open; an infinite lower bound leaves no time (CORALean: finite lb)
    if (std::isinf(lb) || lb > ub || (lb == ub && !(lc && rc))) return; // the empty interval
    lower_ = lb;
    upper_ = ub;
    lc_ = lc;
    rc_ = std::isinf(ub) ? false : rc;
    empty_ = false;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
