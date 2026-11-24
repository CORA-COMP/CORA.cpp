// contains - whether a time interval contains a number or another time interval
//
// Syntax:   bool in = I.contains(t, tol);   bool in = I.contains(S, tol);
// Inputs:   t or S - a number or an interval
//           tol - the tolerance of the bounds
// Outputs:  in - whether it lies in the interval
// See also: stl_interval.h
//
// CORA: specification/@stlInterval/contains_.m
// CORA authors: Florian Lercher
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/specification/stl_interval.h"

#include "cora/global/within_tol.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool StlInterval::contains(double t, Tol tol) const {
    detail::Entry entry;
    const double tl = resolve_tol(entry, tol, "contains");
    if (empty_) return false;
    return (lower_ < t && t < upper_) || (lc_ && within_tol(lower_, t, tl)) ||
           (rc_ && within_tol(upper_, t, tl));
}

bool StlInterval::contains(const StlInterval &S, Tol tol) const {
    detail::Entry entry;
    const double tl = resolve_tol(entry, tol, "contains");
    if (empty_) return S.empty_;
    if (S.empty_) return true;
    const bool upper_included =
        upper_ > S.upper_ || (within_tol(upper_, S.upper_, tl) && (rc_ || !S.rc_));
    const bool lower_included =
        lower_ < S.lower_ || (within_tol(lower_, S.lower_, tl) && (lc_ || !S.lc_));
    return lower_included && upper_included;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
