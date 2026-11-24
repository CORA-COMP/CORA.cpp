// minimum_time - the earliest time a formula looks at
//
// Syntax:   t = phi.minimum_time();
// Inputs:   none
// Outputs:  t - the sum of the window starts along the shallowest path
// See also: stl.h
//
// CORA: specification/@stl/minimumTime.m
// CORA authors: Florian Lercher
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/specification/stl.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// aux_recursive(obj, time): the earliest time below the node, the time already passed being time.
double aux_earliest(const Stl &phi, double time) {
    const StlKind k = phi.kind();
    // the operators that move the time on, in the order of CORA's branches
    if (k == StlKind::next) return aux_earliest(phi.lhs(), phi.from() + time);
    if (k == StlKind::not_) return aux_earliest(phi.lhs(), time);
    if (k == StlKind::and_ || k == StlKind::or_)
        return std::min(aux_earliest(phi.lhs(), time), aux_earliest(phi.rhs(), time));
    if (k == StlKind::finally || k == StlKind::globally) return phi.from() + time;
    return time;
}

} // namespace


// ===========================================  MAIN  =========================================== //

double Stl::minimum_time() const {
    detail::Entry entry;
    return aux_earliest(*this, 0.0);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
