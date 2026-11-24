// plus - the sum of a time interval and another interval, an Interval or a number
//
// Syntax:   K = I.plus(J);   K = I.plus(c);   K = I.plus(Interval)
// Inputs:   other - an interval, an Interval or a number
// Outputs:  K - the sum
// See also: stl_interval.h
//
// CORA: specification/@stlInterval/plus.m
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

// The lower bound of a sum of intervals stays at least zero (CORALean: max(0, lb1 + lb2)).
StlInterval StlInterval::plus(const StlInterval &other) const {
    detail::Entry entry;
    if (empty_ || other.empty_) return {};
    return {std::max(0.0, lower_ + other.lower_), upper_ + other.upper_, lc_ && other.lc_,
            rc_ && other.rc_};
}

StlInterval StlInterval::plus(double c) const {
    detail::Entry entry;
    if (empty_) return {};
    // a negative lower bound is CORA's constructor error (the checks of nested calls are off)
    CORA_CHECK(lower_ + c >= 0, Error::wrong_input_in_constructor,
               [] { return std::string("plus: the lower bound must stay non-negative"); });
    return {lower_ + c, upper_ + c, lc_, rc_};
}

StlInterval StlInterval::plus(const Interval &J) const {
    detail::Entry entry;
    CORA_CHECK(J.inf().numel() == 1, Error::dimension_mismatch,
               [] { return std::string("plus: the Interval must be one-dimensional"); });
    const double inf = J.inf().item(), sup = J.sup().item();
    if (empty_ || sup < inf) return {};
    CORA_CHECK(lower_ + inf >= 0, Error::wrong_input_in_constructor,
               [] { return std::string("plus: the lower bound must stay non-negative"); });
    return {lower_ + inf, upper_ + sup, lc_, rc_};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
