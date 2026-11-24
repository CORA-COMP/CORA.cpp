// get_times - the times at which the identified parts of a formula are evaluated
//
// Syntax:   times = phi.get_times();
// Inputs:   none
// Outputs:  times - for each identifier the times as points (time_point) or windows (time_int)
// See also: stl.h
//
// CORA: specification/@stl/getTimes.m
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

/// The time a node is looked at: a point, or a window once an operator with a window passed.
struct Time {
    bool point;
    double t;
    StlInterval window;
};

/// The times found for each identifier, in the order of the walk.
struct Collected {
    std::vector<std::vector<double>> points;
    std::vector<std::vector<StlInterval>> windows;
};

/// The time moved by a (next): a point moves, a window shifts.
Time aux_shifted(const Time &time, double a) {
    return time.point ? Time{true, a + time.t, StlInterval()}
                      : Time{false, 0.0, time.window.plus(a)};
}

/// The time under an operator with the window w: the window plus the time (always a window).
Time aux_under(const StlInterval &w, const Time &time) {
    return {false, 0.0, time.point ? w.plus(time.t) : w.plus(time.window)};
}

/// aux_recursive(obj, time): collects the times of the identified atomic parts below the node.
void aux_walk(const Stl &phi, const Time &time, Collected &out) {
    const bool logic = phi.kind() == StlKind::and_ || phi.kind() == StlKind::or_;
    // & and | without an id pass the time to both operands
    if (phi.id() == 0 && logic) {
        aux_walk(phi.lhs(), time, out);
        aux_walk(phi.rhs(), time, out);
    } else if (!phi.temporal()) {
        // an identified part records the time it is looked at
        if (phi.id() == 0) return;
        const std::size_t k = static_cast<std::size_t>(phi.id());
        if (time.point) {
            if (out.points.size() < k) out.points.resize(k);
            out.points[k - 1].push_back(time.t);
        } else {
            if (out.windows.size() < k) out.windows.resize(k);
            out.windows[k - 1].push_back(time.window);
        }
    } else if (phi.kind() == StlKind::next) {
        // next shifts the time
        aux_walk(phi.lhs(), aux_shifted(time, phi.from()), out);
    } else if (phi.kind() == StlKind::until || phi.kind() == StlKind::release) {
        const Time inner = aux_under(phi.interval(), time);
        aux_walk(phi.lhs(), inner, out);
        aux_walk(phi.rhs(), inner, out);
    } else {
        // finally, globally; CORA stops with an unassigned result at a temporal ~, we look inside
        aux_walk(phi.lhs(), phi.kind() == StlKind::not_ ? time : aux_under(phi.interval(), time),
                 out);
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

StlTimes Stl::get_times() const {
    detail::Entry entry;
    Collected out;
    aux_walk(*this, Time{true, 0.0, StlInterval()}, out);
    // both lists get the length of the longer one
    const std::size_t n = std::max(out.points.size(), out.windows.size());
    out.points.resize(n);
    out.windows.resize(n);
    return {out.windows, out.points};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
