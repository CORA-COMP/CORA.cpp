// verify - verifies a specification of a neural network controlled system, as CORA's verify
//
// First random runs are simulated; if one violates the specification, the reach is computed from
// its start point alone (without the specification) and the run is FALSIFIED when that reach does
// not verify the specification either, as in CORA. Otherwise the reach from the whole initial set
// stops at the first violating step; VERIFIED if it completes and the specification holds on its
// sets, else UNKNOWN. The recursive splitting of the initial set (splitR0) is not ported.
//
// Syntax:   v = sys.verify(X0, U, t_final, options, spec, points, rng);
// Inputs:   X0 - the initial set (Zonotope or PolyZonotope), U - the external input set (optional)
//           spec - a timed specification, points - the number of random runs
// Outputs:  v - the verdict, the start of a violating run, and the number of runs
// See also: reach, simulate_random
//
// CORA: contDynamics/@neurNetContrSys/verify.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/neur_net_contr_sys/private/priv_holds.h"
#include "cora/cont_dynamics/neur_net_contr_sys/neur_net_contr_sys.h"
#include "global/format.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The first run that violates the specification at a sampled time: its index (-1 if none) and
/// the sample.
std::pair<int64_t, std::size_t> aux_violating_run(const TimedSpecification &spec,
                                                  const ClosedLoopTrajectories &traj) {
    for (std::size_t k = 0; k < traj.x.size(); ++k) {
        if (!priv_in_window(spec, traj.time[k], traj.time[k])) continue;
        const Tensor &x = traj.x[k];
        for (int64_t j = 0; j < x.shape()[1]; ++j)
            if (!spec.spec.check(Zonotope(x.slice(1, j, j + 1), x.zeros_like({x.shape()[0], 0}))))
                return {j, k};
    }
    return {-1, 0};
}

/// The enclosing zonotope of a set.
Zonotope aux_zonotope(const Zonotope &Z) { return Z; }
Zonotope aux_zonotope(const PolyZonotope &P) { return P.zonotope(); }

/// Whether a run of the reach completed and its sets satisfy the specification.
template <class S>
bool aux_verified(const TimedSpecification &spec, const ClosedLoopReach<S> &R) {
    return R.completed && priv_holds_reach(spec, R, [](const S &s) { return aux_zonotope(s); });
}

template <class S>
ClosedLoopVerdict aux_verify(const NeurNetContrSys &sys, const S &X0, const Zonotope &sim_set,
                             const std::optional<Zonotope> &U, double t_final,
                             const ClosedLoopOptions &options, const TimedSpecification &spec,
                             int64_t points, Rng &rng) {
    ClosedLoopVerdict out;
    out.simulations = static_cast<int>(points);
    // random runs, sampled at the steps of the reach
    const ClosedLoopTrajectories traj = sys.simulate_random(
        sim_set, U, t_final, std::min(sys.dt(), options.plant.time_step), points, 0.5, rng);
    const auto [violating, sample] = aux_violating_run(spec, traj);
    if (violating >= 0) {
        out.violation_time = traj.time[sample];
        out.violating_state = traj.x[sample].slice(1, violating, violating + 1);
        // the reach from the start of the violating run, without the specification
        const Tensor start = traj.x[0].slice(1, violating, violating + 1);
        out.violating_start = start;
        const PolyZonotope point{Zonotope(start, start.zeros_like({sys.dim(), 0}))};
        const ClosedLoopReach<PolyZonotope> R = sys.reach(point, U, t_final, options);
        out.verdict = aux_verified(spec, R) ? Verdict::verified : Verdict::falsified;
        return out;
    }
    // the whole initial set, stopping at a violation
    const ClosedLoopReach<S> R = sys.reach(X0, U, t_final, options, spec);
    out.verdict = aux_verified(spec, R) ? Verdict::verified : Verdict::unknown;
    out.time_reached = R.time.back();
    out.exploded = R.exploded;
    out.violated = R.violated;
    return out;
}

} // namespace


// ===========================================  MAIN  =========================================== //

ClosedLoopVerdict NeurNetContrSys::verify(const PolyZonotope &X0, const std::optional<Zonotope> &U,
                                          double t_final, const ClosedLoopOptions &options,
                                          const TimedSpecification &spec, int64_t points,
                                          Rng &rng) const {
    detail::Entry entry;
    CORA_CHECK(points >= 1, Error::wrong_value,
               [] { return std::string("verify: needs at least one simulated run"); });
    return aux_verify(*this, X0, X0.zonotope(), U, t_final, options, spec, points, rng);
}

ClosedLoopVerdict NeurNetContrSys::verify(const Zonotope &X0, const std::optional<Zonotope> &U,
                                          double t_final, const ClosedLoopOptions &options,
                                          const TimedSpecification &spec, int64_t points,
                                          Rng &rng) const {
    detail::Entry entry;
    CORA_CHECK(points >= 1, Error::wrong_value,
               [] { return std::string("verify: needs at least one simulated run"); });
    return aux_verify(*this, X0, X0, U, t_final, options, spec, points, rng);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
