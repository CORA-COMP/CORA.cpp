// priv_holds - whether a set satisfies a timed specification
//
// CORA's check(spec, S, time) of a set: a specification with a time window applies to the sets
// whose time interval meets it (up to 1e-8); one without applies everywhere. The set of a
// specification is a Specification, whose check runs the set's own operations. A polynomial
// zonotope is checked on its zonotope enclosure.
//
// Syntax:   bool ok = priv_holds(spec, S, t0, t1);
// See also: neur_net_contr_sys.h, reach.cpp, verify.cpp
//
// CORA: specification/@specification/check.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_dynamics/neur_net_contr_sys/neur_net_contr_sys.h"

#include "cora/global/tensor/tolerance.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// Whether the window of the specification meets [t0, t1].
inline bool priv_in_window(const TimedSpecification &spec, double t0, double t1) {
    constexpr double tol = numerics::kTimeStepMatch;
    return !spec.time || (spec.time->first <= t1 + tol && t0 <= spec.time->second + tol);
}

/// Whether the set over [t0, t1] satisfies the specification (true where it does not apply).
inline bool priv_holds(const TimedSpecification &spec, const Zonotope &S, double t0, double t1) {
    return !priv_in_window(spec, t0, t1) || spec.spec.check(S);
}

/// Whether the reachable sets satisfy the specification (CORA's check(spec, reachSet)); a window
/// that no step meets is violated, as the behavior there is undefined.
template <class S, class Enclose>
bool priv_holds_reach(const TimedSpecification &spec, const ClosedLoopReach<S> &R,
                      Enclose to_zonotope) {
    constexpr double tol = numerics::kTimeStepMatch;
    bool covered = !spec.time;
    for (std::size_t j = 0; j < R.time_interval.size(); ++j) {
        const double t0 = R.time[j], t1 = R.time[j + 1];
        bool ok = true;
        if (spec.time && spec.time->first == spec.time->second) {
            // an instant is checked at the start or the end of the step, else inside it
            const double t = spec.time->first;
            if (std::abs(t - t0) <= tol)
                ok = spec.spec.check(to_zonotope(R.time_point[j]));
            else if (std::abs(t - t1) <= tol)
                ok = spec.spec.check(to_zonotope(R.time_point[j + 1]));
            else
                ok = priv_holds(spec, to_zonotope(R.time_interval[j]), t0, t1);
        } else {
            ok = priv_holds(spec, to_zonotope(R.time_interval[j]), t0, t1);
        }
        covered = covered || priv_in_window(spec, t0, t1);
        if (!ok) return false;
    }
    return covered;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
