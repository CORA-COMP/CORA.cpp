// priv_location_reach - the reachable sets of one location and its jumps, as CORA's location.reach
//
// The flow x' = A x + B u + c runs as the canonical system x' = A x + u with the input set
// B U + c, step by step with the algorithm "standard", until a time-interval set leaves the
// invariant or t_final is reached. Every time-interval set is checked against the specifications
// (CORA checks inside linearSys.reach and the last set after it); a violation ends the analysis.
// The guards that the sets meet are intersected and each intersection is mapped by its reset.
// A start time [t0, t1] computes from t0 and widens the times of the sets by t1 - t0.
//
// Syntax:   res = priv_location_reach(loc, index, R0, t_start, t_final, U, dt, alg, specs, final);
// Inputs:   loc, index - the location and its number
//           R0 - the initial set (a zonotope or a bundle)
//           t_start, t_final - the start times and the time horizon
//           U - the input set of the flow's B, optional
//           dt - the step size of the location
//           alg - the algorithm (taylor_terms, zonotope_order, the guard intersection)
//           specs - the specifications active in the location
//           final - locations that are not entered
// Outputs:  res - the reachable sets, the jumps, and whether the specifications held
// See also: priv_pot_int, priv_guard_intersect, HybridAutomaton::reach
//
// CORA: hybridDynamics/@location/reach.m
// CORA authors: Matthias Althoff, Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv.h"
#include "cora/cont_set/zono_bundle/zono_bundle.h"
#include "cora/global/within_tol.h"
#include "hybrid_dynamics/location/private/priv.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The flow ----------------------------------------------------------------------------------------

/// The steps of the location, one stepper per zonotope of the initial set (a bundle propagates
/// zonotope by zonotope, as CORA's operations on bundles do).
class BundleSteps {
  public:
    BundleSteps(const LinearSys &canonical, const Set &R0, const Zonotope &U_state, double dt,
                const alg::HybridStandard &alg) {
        const Tensor e_adt = (canonical.A() * dt).expm();
        const Interval F = canonical.correction_matrix_state(dt, alg.taylor_terms);
        const InputSolution in = canonical.input_solution(U_state, dt, alg.taylor_terms);
        for (const Zonotope &Z : priv_zonotopes(R0))
            steps_.emplace_back(Z, e_adt, F, in, alg.zonotope_order);
    }

    /// The time-interval set of the next step; time_point() is then its end.
    Set next() {
        std::vector<Zonotope> parts;
        parts.reserve(steps_.size());
        for (StandardSteps &s : steps_) parts.push_back(s.next());
        return priv_bundle(std::move(parts));
    }

    Set time_point() const {
        std::vector<Zonotope> parts;
        parts.reserve(steps_.size());
        for (const StandardSteps &s : steps_) parts.push_back(s.time_point());
        return priv_bundle(std::move(parts));
    }

  private:
    std::vector<StandardSteps> steps_;
};

/// Whether every specification holds for the set.
bool aux_specs_hold(const std::vector<Specification> &specs, const Set &S) {
    for (const Specification &spec : specs)
        if (!spec.check(S)) return false;
    return true;
}

/// The continuous part: the sets from t0 until they leave the invariant or reach t_final; false
/// in the second member if a specification is violated (the sets end with the violating one).
std::pair<LocationReachSet, bool> aux_flow(const Location &loc, int64_t index, const Set &R0,
                                           double t0, double t_final, const Zonotope &U_state,
                                           double dt, const alg::HybridStandard &alg,
                                           const std::vector<Specification> &specs) {
    LocationReachSet R;
    R.location = index;
    R.time_point.push_back(R0);
    R.time_point_times.push_back({t0, t0});
    // CORA: tStart = tFinal or an empty invariant leave the start set alone
    if (within_tol(t0, t_final, numerics::kTimeRound) || loc.invariant().representsa("emptySet"))
        return {R, true};
    const int steps = static_cast<int>(std::floor((t_final - t0) / dt + numerics::kStepCount));
    const LinearSys canonical(loc.flow().A(), loc.flow().A().eye_like());
    BundleSteps step(canonical, R0, U_state, dt, alg);
    for (int k = 0; k < steps; ++k) {
        // one step: the enclosure of [t_k, t_k+1], then the set at t_k+1
        const Set ti = step.next();
        R.time_interval.push_back(ti);
        R.time_interval_times.push_back({t0 + k * dt, t0 + (k + 1) * dt});
        R.time_point.push_back(step.time_point());
        R.time_point_times.push_back({t0 + (k + 1) * dt, t0 + (k + 1) * dt});
        if (!aux_specs_hold(specs, ti)) return {R, false};
        if (!priv_meets(loc.invariant(), ti)) break; // the set has left the invariant
    }
    return {R, true};
}

// The specifications ------------------------------------------------------------------------------

/// The specifications that apply: an unsafe set only if it meets the invariant (CORA).
std::vector<Specification> aux_active(const Location &loc,
                                      const std::vector<Specification> &specs) {
    std::vector<Specification> out;
    for (const Specification &spec : specs) {
        if (spec.type() == SpecType::unsafe_set) {
            std::vector<Tensor> rows, offsets;
            for (const Halfspace &h : spec.halfspaces()) {
                rows.push_back(h.a.matrix_transpose());
                offsets.push_back(h.a.full_like({1, 1}, h.b));
            }
            const Polytope P(Tensor::concat(rows, -2), Tensor::concat(offsets, -2));
            if (!loc.invariant().is_intersecting(P)) continue;
        }
        out.push_back(spec);
    }
    return out;
}

} // namespace


// ===========================================  MAIN  =========================================== //

Zonotope priv_state_input(const Location &loc, const std::optional<Zonotope> &U) {
    const int64_t n = loc.dim();
    if (!U || !loc.flow().B()) return {loc.c(), loc.c().zeros_like({n, 0})};
    const Tensor &B = *loc.flow().B();
    return {B.matmul(U->center()) + loc.c(), B.matmul(U->generators())};
}

LocationResult priv_location_reach(const Location &loc, int64_t index, const Set &R0,
                                   const TimeRange &t_start, double t_final,
                                   const std::optional<Zonotope> &U, double time_step,
                                   const alg::HybridStandard &alg,
                                   const std::vector<Specification> &specs,
                                   const std::vector<int64_t> &final_locations) {
    const Zonotope U_state = priv_state_input(loc, U);
    auto [R, verified] = aux_flow(loc, index, R0, t_start[0], t_final, U_state, time_step, alg,
                                  aux_active(loc, specs));
    LocationResult out{std::move(R), {}, verified};
    if (!verified) return out; // the analysis ends at a violation

    // the guards the sets meet, their intersections, and the resets of those
    const GuardHits hits = priv_pot_int(loc, out.R, final_locations);
    const double widen = t_start[1] - t_start[0];
    for (const GuardIntersection &g :
         priv_guard_intersect(loc, hits, out.R, U_state, alg.guard_intersect)) {
        const Transition &trans = loc.transitions()[std::size_t(g.transition)];
        const std::vector<TimeRange> &times =
            hits.time_point ? out.R.time_point_times : out.R.time_interval_times;
        const TimeRange time{times[std::size_t(g.first)][0],
                             times[std::size_t(g.last)][1] + (hits.time_point ? 0 : widen)};
        out.jumps.push_back({trans.reset().evaluate(*g.set, U), trans.target(), time});
    }
    // an uncertain start time widens the times of the sets (CORA's updateTime)
    for (TimeRange &t : out.R.time_point_times) t[1] += widen;
    for (TimeRange &t : out.R.time_interval_times) t[1] += widen;
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
