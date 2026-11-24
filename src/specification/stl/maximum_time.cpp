// maximum_time - the latest time a formula looks at
//
// Syntax:   t = phi.maximum_time();
// Inputs:   none
// Outputs:  t - the sum of the window ends along the deepest path
// See also: stl.h
//
// CORA: specification/@stl/maximumTime.m
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

/// aux_recursive(obj, time): the latest time below the node, the time already passed being time.
double aux_latest(const Stl &phi, double time) {
    const StlKind k = phi.kind();
    // the operators that move the time on, in the order of CORA's branches
    if (k == StlKind::next) return aux_latest(phi.lhs(), phi.from() + time);
    if (k == StlKind::not_) return aux_latest(phi.lhs(), time);
    if (k == StlKind::and_ || k == StlKind::or_)
        return std::max(aux_latest(phi.lhs(), time), aux_latest(phi.rhs(), time));
    if (k == StlKind::finally || k == StlKind::globally)
        return phi.to() + aux_latest(phi.lhs(), time);
    if (k == StlKind::until || k == StlKind::release)
        return phi.to() + std::max(aux_latest(phi.lhs(), time), aux_latest(phi.rhs(), time));
    return time;
}

} // namespace


// ===========================================  MAIN  =========================================== //

double Stl::maximum_time() const {
    detail::Entry entry;
    return aux_latest(*this, 0.0);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
