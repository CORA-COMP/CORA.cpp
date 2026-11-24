// minus - the difference of a time interval and another interval, an Interval or a number
//
// Syntax:   K = I.minus(J);   K = I.minus(c);   K = I.minus(Interval)
// Inputs:   other - an interval, an Interval or a number
// Outputs:  K - the difference, lower bound at least 0
// See also: stl_interval.h
//
// CORA: specification/@stlInterval/minus.m
// CORA authors: Florian Lercher
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/specification/stl_interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The interval [lb, ub] with the lower bound raised to 0 (closed there) if it is negative.
StlInterval aux_clamped(double lb, double ub, bool lc, bool rc) {
    if (lb < 0) return {0, ub, true, rc};
    return {lb, ub, lc, rc};
}

} // namespace


// ===========================================  MAIN  =========================================== //


StlInterval StlInterval::minus(const StlInterval &other) const {
    detail::Entry entry;
    if (empty_ || other.empty_) return {};
    // lower - upper' (inf if upper' is) to upper - lower'
    return aux_clamped(lower_ - other.upper_, upper_ - other.lower_, lc_ && other.rc_,
                       rc_ && other.lc_);
}

StlInterval StlInterval::minus(double c) const {
    detail::Entry entry;
    if (empty_) return {};
    return aux_clamped(lower_ - c, upper_ - c, lc_, rc_);
}

StlInterval StlInterval::minus(const Interval &J) const {
    detail::Entry entry;
    CORA_CHECK(J.inf().numel() == 1, Error::dimension_mismatch,
               [] { return std::string("minus: the Interval must be one-dimensional"); });
    const double inf = J.inf().item(), sup = J.sup().item();
    if (empty_ || sup < inf) return {};
    return aux_clamped(lower_ - sup, upper_ - inf, lc_, rc_);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
