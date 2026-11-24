// isequal - whether two time intervals are equal up to a tolerance
//
// Syntax:   bool same = I.isequal(J, tol);   I == J
// Inputs:   other - an interval
//           tol - the tolerance of the bounds
// Outputs:  same - whether they are equal
// See also: stl_interval.h
//
// CORA: specification/@stlInterval/isequal.m
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

bool StlInterval::isequal(const StlInterval &other, Tol tol) const {
    detail::Entry entry;
    const double tl = resolve_tol(entry, tol, "isequal");
    if (empty_ || other.empty_) return empty_ && other.empty_;
    return within_tol(lower_, other.lower_, tl) && within_tol(upper_, other.upper_, tl) &&
           lc_ == other.lc_ && rc_ == other.rc_;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
