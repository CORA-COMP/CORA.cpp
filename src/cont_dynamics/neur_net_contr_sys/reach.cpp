// reach - the reachable sets of a neural network controlled system, as CORA's neurNetContrSys.reach
//
// Per control period [t, t + dt]: the network is enclosed over the current set X together with its
// input, (x, N(x)) (NeuralNetwork::evaluate_with_input, CORA's evaluate_ and aux_appendU2X), and
// the extended system is reached from that set with init_reach and post of the plant's algorithm,
// the step size cut to the period (the last step of a period may be shorter, as CORA's tVec; the
// others all of the size time_step, which CORA's differences of tVec only approximate). The
// first step is reduced to zonotope_order as CORA's aux_reachability does. The sets are projected
// to the plant's states; the set at the end of the period starts the next. After every step the
// set over it is checked against the specification, and the run stops at a violation, or when
// the sets of a step explode (CORA's setexplosion, which includes a range that leaves the domain of
// the dynamics' functions), returning the sets so far.
//
// Syntax:   R = sys.reach(X0, U, t_final, {.plant = {.time_step = 0.1}}, spec);
// Inputs:   X0 - the initial set (Zonotope or PolyZonotope), U - the external input set (optional)
//           t_final - the horizon, options - the plant's alg::Linearization and NnOptions
//           spec - a timed specification (optional)
// Outputs:  R - the sets over and at the steps, in the states of the plant
// See also: simulate, verify, NonlinearSys::init_reach
//
// CORA: contDynamics/@neurNetContrSys/reach.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv.h"
#include "cont_dynamics/neur_net_contr_sys/private/priv_holds.h"
#include "cora/cont_dynamics/neur_net_contr_sys/neur_net_contr_sys.h"
#include "global/format.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The time grid -----------------------------------------------------------------------------------

/// The times t0, t0 + h, ..., t1 with a shorter last step if h does not divide t1 - t0 (the
/// tVec of CORA's reach).
std::vector<double> aux_grid(double t0, double t1, double h) {
    const int steps = priv_num_steps(t1 - t0, h);
    std::vector<double> t;
    t.reserve(std::size_t(steps) + 1);
    for (int i = 0; i < steps; ++i) t.push_back(t0 + i * h);
    t.push_back(t1);
    return t;
}

/// The size of step i: h itself, so that a period of whole steps has equal steps and no drift of
/// the differences of t0 + i h (as CORALean's N steps of one size); a shorter last step only if h
/// does not divide the period.
double aux_step(const std::vector<double> &t, std::size_t i, double h) {
    const double rest = t[i + 1] - t[i];
    return std::abs(rest - h) <= numerics::kStepCount * h ? h : rest;
}

// The sets of the extended system -----------------------------------------------------------------

/// The first n states of a set of the extended system (CORA's project(R, 1:n)).
Zonotope aux_project(const Zonotope &Z, int64_t n) {
    return {Z.center().slice(-2, 0, n), Z.generators().slice(-2, 0, n)};
}

PolyZonotope aux_project(const PolyZonotope &P, int64_t n) {
    return {P.center().slice(0, 0, n), P.generators().slice(0, 0, n),
            P.independent_generators().slice(0, 0, n), P.exponents(), P.ids()};
}

/// The set of the pairs (x, N(x)) for x in X.
Zonotope aux_joint(const NeuralNetwork &nn, const Zonotope &X, const NnOptions &options) {
    return nn.evaluate_with_input(X, options);
}

PolyZonotope aux_joint(const NeuralNetwork &nn, const PolyZonotope &X, const NnOptions &options) {
    return nn.evaluate_with_input(X, options);
}

/// The enclosing zonotope of a set, for the checks.
Zonotope aux_zonotope(const Zonotope &Z) { return Z; }
Zonotope aux_zonotope(const PolyZonotope &P) { return P.zonotope(); }

// The steps of the extended system ----------------------------------------------------------------

/// The first step, with both sets reduced as CORA's aux_reachability does.
NonlinearStep aux_first(const NonlinearSys &sys, const Zonotope &R0,
                        const std::optional<Zonotope> &U, const alg::Linearization &alg) {
    NonlinearStep step = sys.init_reach(R0, U, alg);
    step.time_interval = step.time_interval.reduce(alg.zonotope_order);
    step.time_point = step.time_point.reduce(alg.zonotope_order);
    return step;
}

PolyNonlinearStep aux_first(const NonlinearSys &sys, const PolyZonotope &R0,
                            const std::optional<Zonotope> &U, const alg::Linearization &alg) {
    PolyNonlinearStep step = sys.init_reach(R0, U, alg);
    step.time_interval = step.time_interval.reduce(alg.zonotope_order);
    step.time_point = step.time_point.reduce(alg.zonotope_order);
    return step;
}

/// The step after `step` (post reduces).
NonlinearStep aux_next(const NonlinearSys &sys, const NonlinearStep &step,
                       const std::optional<Zonotope> &U, const alg::Linearization &alg) {
    return sys.post(step, U, alg);
}

PolyNonlinearStep aux_next(const NonlinearSys &sys, const PolyNonlinearStep &step,
                           const std::optional<Zonotope> &U, const alg::Linearization &alg) {
    return sys.post(step, U, alg);
}

// The loop ----------------------------------------------------------------------------------------

/// The checks of the problem, for either kind of set.
template <class S>
void aux_validate(const NeurNetContrSys &sys, const S &X0, const std::optional<Zonotope> &U,
                  double t_final, const ClosedLoopOptions &options) {
    detail::Validation entry;
    CORA_CHECK(t_final > 0 && t_final < 1e300, Error::wrong_value, [&] {
        return "reach: t_final must be positive and finite, got " + std::to_string(t_final);
    });
    CORA_CHECK(X0.dim() == sys.dim(), Error::dimension_mismatch, [&] {
        return "reach: the initial set has the dimension " + std::to_string(X0.dim()) +
               ", the system " + std::to_string(sys.dim());
    });
    CORA_CHECK(bool(U) == (sys.num_external_inputs() > 0), Error::wrong_value, [&] {
        return "reach: the system has " + std::to_string(sys.num_external_inputs()) +
               " external inputs; the input set must be given exactly then";
    });
    options.plant.validate();
}

/// One control period: the network over X together with its input, then the steps of the period
/// (the last may be shorter), each projected to the plant's states and checked. Returns false at
/// a violation.
template <class S, class Step>
bool aux_period(const NeurNetContrSys &sys, const S &X, const std::optional<Zonotope> &U,
                const std::vector<double> &period, const ClosedLoopOptions &options,
                const TimedSpecifications &specs, ClosedLoopReach<S> &R) {
    const S R0 = aux_joint(sys.network(), X, options.nn);
    const std::vector<double> inner = aux_grid(period[0], period[1], options.plant.time_step);
    alg::Linearization alg = options.plant;
    std::optional<Step> step;
    for (std::size_t i = 0; i + 1 < inner.size(); ++i) {
        // the first step starts from the joint set, the others from the step before
        alg.time_step = aux_step(inner, i, options.plant.time_step);
        step = i == 0 ? aux_first(sys.closed_loop(), R0, U, alg)
                      : aux_next(sys.closed_loop(), *step, U, alg);
        R.time_interval.push_back(aux_project(step->time_interval, sys.dim()));
        R.time_point.push_back(aux_project(step->time_point, sys.dim()));
        R.time.push_back(inner[i + 1]);
        // the set over the step against the specification
        for (const TimedSpecification &spec : specs)
            if (!priv_holds(spec, aux_zonotope(R.time_interval.back()), inner[i], inner[i + 1]))
                return false;
    }
    return true;
}

/// The loop over the control periods, for either kind of set.
template <class S, class Step>
ClosedLoopReach<S> aux_reach(const NeurNetContrSys &sys, S X, const std::optional<Zonotope> &U,
                             double t_final, const ClosedLoopOptions &options,
                             const TimedSpecifications &specs) {
    aux_validate(sys, X, U, t_final, options);
    ClosedLoopReach<S> R;
    R.time = {0.0};
    R.time_point.push_back(X);
    const std::vector<double> periods = aux_grid(0.0, t_final, sys.dt());
    try {
        for (std::size_t p = 0; p + 1 < periods.size(); ++p) {
            if (!aux_period<S, Step>(sys, X, U, {periods[p], periods[p + 1]}, options, specs, R)) {
                R.violated = true;
                R.completed = false;
                return R;
            }
            // the set at the end of the period starts the next
            X = R.time_point.back();
        }
    } catch (const Error &e) {
        // the sets so far are returned when they explode, or leave a domain (tan at a pole)
        if (e.id() != Error::reach_set_explosion && e.id() != Error::out_of_domain)
            std::rethrow_exception(std::current_exception());
        R.exploded = true;
        R.completed = false;
    }
    return R;
}

} // namespace


// ===========================================  MAIN  =========================================== //

ClosedLoopReach<Zonotope>
NeurNetContrSys::reach(const Zonotope &X0, const std::optional<Zonotope> &U, double t_final,
                       const ClosedLoopOptions &options,
                       const std::optional<TimedSpecification> &spec) const {
    detail::Entry entry;
    // one specification or none is a list of at most one
    return reach(X0, U, t_final, options,
                 spec ? TimedSpecifications{*spec} : TimedSpecifications{});
}

ClosedLoopReach<PolyZonotope>
NeurNetContrSys::reach(const PolyZonotope &X0, const std::optional<Zonotope> &U, double t_final,
                       const ClosedLoopOptions &options,
                       const std::optional<TimedSpecification> &spec) const {
    detail::Entry entry;
    // one specification or none is a list of at most one
    return reach(X0, U, t_final, options,
                 spec ? TimedSpecifications{*spec} : TimedSpecifications{});
}

// The lists of specifications ---------------------------------------------------------------------

ClosedLoopReach<Zonotope> NeurNetContrSys::reach(const Zonotope &X0,
                                                 const std::optional<Zonotope> &U, double t_final,
                                                 const ClosedLoopOptions &options,
                                                 const TimedSpecifications &specs) const {
    detail::Entry entry;
    return aux_reach<Zonotope, NonlinearStep>(*this, X0, U, t_final, options, specs);
}

ClosedLoopReach<PolyZonotope>
NeurNetContrSys::reach(const PolyZonotope &X0, const std::optional<Zonotope> &U, double t_final,
                       const ClosedLoopOptions &options, const TimedSpecifications &specs) const {
    detail::Entry entry;
    return aux_reach<PolyZonotope, PolyNonlinearStep>(*this, X0, U, t_final, options, specs);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
