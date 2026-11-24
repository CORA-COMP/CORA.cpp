// representsa - whether a time interval is a set of the given type
//
// Syntax:   bool is = I.representsa("point", tol);
// Inputs:   type - the set type
//           tol - the tolerance
// Outputs:  is - whether it represents that type
// See also: stl_interval.h
//
// CORA: specification/@stlInterval/representsa_.m
// CORA authors: Florian Lercher
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/specification/stl_interval.h"

#include "cora/global/within_tol.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

// CORA forwards the other types to the closed Interval; those are Error::not_supported here.
bool StlInterval::representsa(const std::string &type, Tol tol) const {
    detail::Entry entry;
    const double tl = resolve_tol(entry, tol, "representsa");
    if (type == "emptySet") return empty_;
    if (type == "stlInterval") return true;
    CORA_CHECK(type == "point" || type == "origin", Error::not_supported,
               [&] { return "representsa: the type '" + type + "' is not supported"; });
    if (empty_) return false;
    if (!(lc_ && rc_)) return false; // a half-open interval is not a closed set
    if (type == "origin") return within_tol(lower_, 0.0, tl) && within_tol(upper_, 0.0, tl);
    return !std::isinf(upper_) && within_tol((upper_ - lower_) / 2, 0.0, tl);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
