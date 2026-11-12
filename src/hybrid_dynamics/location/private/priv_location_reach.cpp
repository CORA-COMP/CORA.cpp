// priv_location_reach - the reachable sets of one location and its jumps, as CORA's location.reach
//
// The flow x' = A x + B u + c runs as the canonical system x' = A x + u with the input set
// B U + c, step by step with the algorithm "standard", until a time-interval set leaves the
// invariant or t_final is reached; a LinParamSys flow runs CORA's initReach and post with the
// center of its input set as the constant input, a NonlinearSys flow its init_reach and post
// (alg "lin") with the input set U. Every time-interval set is checked against the
// specifications (CORA checks inside linearSys.reach and the last set after it); a violation ends
// the analysis. The guards that the sets meet are intersected and each intersection is mapped by
// its reset. A start time [t0, t1] computes from t0 and widens the times of the sets by t1 - t0.
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
// Last update:   09-November-2026
// Reviewed:      ---

#include "cont_dynamics/lin_param_sys/private/priv.h"
#include "cont_dynamics/linear_sys/private/priv.h"
#include "cont_dynamics/linear_sys/private/priv_float_step.h"
#include "cora/cont_set/registry.h"
#include "cora/cont_set/zono_bundle/zono_bundle.h"
#include "cora/global/within_tol.h"
#include "hybrid_dynamics/location/private/priv.h"

#include <cmath>
#include <memory>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The flow ----------------------------------------------------------------------------------------

/// The steps of the location, one stepper per zonotope of the initial set (a bundle propagates
/// zonotope by zonotope, as CORA's operations on bundles do), by the flow's own algorithm; a
/// nonlinear flow starts from a single zonotope.
class BundleSteps {
  public:
    BundleSteps(const Location &loc, const Set &R0, const std::optional<Zonotope> &U,
                const Zonotope &U_state, double dt, const alg::HybridStandard &alg)
        : start_(priv_zonotopes(R0)) {
        if (const auto *sys = std::get_if<NonlinearSys>(&loc.flow())) {
            // polynomial zonotopes: alg "poly" by choice, alg "lin" for a polynomial start set
            if (alg.polynomialization || R0.is<PolyZonotope>()) {
                alg::PolyNonlinearReach poly_alg = alg::Linearization{dt,
                                                                      alg.taylor_terms,
                                                                      alg.zonotope_order,
                                                                      alg.tensor_order,
                                                                      alg.error_order,
                                                                      alg.intermediate_order};
                if (alg.polynomialization)
                    poly_alg = alg::Polynomialization{dt, alg.taylor_terms, alg.zonotope_order,
                                                      3,  alg.error_order,  alg.intermediate_order};
                // the polynomial start set and the algorithm chosen above
                poly_ = {sys, U, poly_alg, R0.as<PolyZonotope>()};
                return;
            }
            if (start_.size() != 1)
                CORA_FAIL(Error::not_supported,
                          "reach: a nonlinear flow starts from one zonotope, not a bundle; enclose "
                          "the guard intersections with a single method");
            nonlinear_ = {sys,
                          U,
                          {dt, alg.taylor_terms, alg.zonotope_order, alg.tensor_order,
                           alg.error_order, alg.intermediate_order, alg.linear_step}};
            return;
        }
        if (const auto *sys = std::get_if<LinearSys>(&loc.flow())) {
            // CORALean's reachFloatInput on a single zonotope, window by window
            if (alg.linear_step == alg::LinearStep::coralean) {
                if (start_.size() != 1)
                    CORA_FAIL(Error::not_supported,
                              "reach: linear_step = coralean starts from one zonotope, not a "
                              "bundle");
                float_stepper_ = std::make_unique<FloatStepper>(sys->A(), dt, alg.taylor_terms);
                if (!float_stepper_->admissible())
                    CORA_FAIL(Error::wrong_value,
                              "reach: no Taylor order admits the step; shorten the time step");
                float_run_.emplace(float_stepper_->run(U_state, start_[0], alg.zonotope_order));
                return;
            }
            // x' = A x + u with u in B U + c
            const LinearSys canonical(sys->A(), sys->A().eye_like());
            const Tensor e_adt = (canonical.A() * dt).expm();
            const Interval F = canonical.correction_matrix_state(dt, alg.taylor_terms);
            const InputSolution in = canonical.input_solution(U_state, dt, alg.taylor_terms);
            for (const Zonotope &Z : start_)
                steps_.emplace_back(Z, e_adt, F, in, alg.zonotope_order);
            return;
        }
        // a LinParamSys: the input without its center, which is the constant input
        const auto &sys = std::get<LinParamSys>(loc.flow());
        param_alg_ = {dt, alg.taylor_terms, alg.zonotope_order, alg.intermediate_terms, true};
        const int64_t m = sys.B() ? sys.input_dim() : 1;
        const Tensor &c = sys.c();
        Zonotope Uc = U ? *U : Zonotope(c.zeros_like({m, 1}), c.zeros_like({m, 0}));
        const Tensor u_trans = Uc.center();
        Uc = Uc - u_trans;
        mappings_.emplace(priv_mappings(sys, Uc, u_trans, *param_alg_));
    }

    /// The time-interval set of the next step; time_point() is then its end.
    Set next() {
        // polynomial zonotopes first, then the zonotopes of the nonlinear and linear flows
        if (poly_) {
            Poly &f = *poly_;
            poly_step_ = std::visit(
                [&](const auto &a) {
                    return poly_step_ ? f.sys->post(*poly_step_, f.U, a)
                                      : f.sys->init_reach(f.start, f.U, a);
                },
                f.alg);
            return poly_step_->time_interval;
        }
        // a nonlinear flow on a zonotope: init for the first step, post for the others
        if (nonlinear_) {
            const Nonlinear &f = *nonlinear_;
            step_ =
                step_ ? f.sys->post(*step_, f.U, f.alg) : f.sys->init_reach(start_[0], f.U, f.alg);
            return step_->time_interval;
        }
        if (float_run_) return float_run_->next();
        // the linear flows step every zonotope of the bundle
        std::vector<Zonotope> parts;
        parts.reserve(start_.size());
        if (!mappings_) {
            for (StandardSteps &s : steps_) parts.push_back(s.next());
            return priv_bundle(std::move(parts));
        }
        if (param_.empty()) {
            // CORA adds F times the first zonotope of a bundle to every one
            const Zonotope curvature = mappings_->correction * start_.front();
            for (const Zonotope &Z : start_)
                param_.push_back(priv_first_step(Z, curvature, *mappings_, *param_alg_));
        } else {
            for (LinParamStep &step : param_) step = priv_next_step(step, *param_alg_);
        }
        for (const LinParamStep &step : param_) parts.push_back(step.time_interval);
        return priv_bundle(std::move(parts));
    }

    /// The set at the start of the next step (the initial set before the first).
    Set time_point() const {
        if (poly_) return poly_step_ ? poly_step_->time_point : poly_->start;
        if (nonlinear_) return step_ ? step_->time_point : start_[0];
        if (float_run_) return float_run_->time_point();
        std::vector<Zonotope> parts;
        parts.reserve(start_.size());
        if (!mappings_)
            for (const StandardSteps &s : steps_) parts.push_back(s.time_point());
        else if (param_.empty())
            parts = start_;
        else
            for (const LinParamStep &step : param_) parts.push_back(*step.time_point);
        return priv_bundle(std::move(parts));
    }

  private:
    /// A nonlinear flow, its input set and its algorithm.
    struct Nonlinear {
        const NonlinearSys *sys;
        std::optional<Zonotope> U;
        alg::Linearization alg;
    };

    /// A nonlinear flow on polynomial zonotopes, its input set, algorithm and start set.
    struct Poly {
        const NonlinearSys *sys;
        std::optional<Zonotope> U;
        alg::PolyNonlinearReach alg;
        PolyZonotope start;
    };

    std::vector<Zonotope> start_;
    std::optional<Poly> poly_;
    std::optional<PolyNonlinearStep> poly_step_;
    std::optional<Nonlinear> nonlinear_;
    std::optional<NonlinearStep> step_;
    std::vector<StandardSteps> steps_;
    std::unique_ptr<FloatStepper> float_stepper_;
    std::optional<FloatStepper::Run> float_run_;
    std::optional<alg::LinParamStandard> param_alg_;
    std::optional<LinParamMappings> mappings_;
    std::vector<LinParamStep> param_;
};

// The sets of the location ------------------------------------------------------------------------

/// Whether every specification holds for the set.
bool aux_specs_hold(const std::vector<Specification> &specs, const Set &S) {
    for (const Specification &spec : specs)
        if (!spec.check(S)) return false;
    return true;
}

/// The continuous part: the sets from t0 until they leave the invariant or reach t_final; false
/// in the second member if a specification is violated (the sets end with the violating one).
std::pair<LocationReachSet, bool>
aux_flow(const Location &loc, int64_t index, const Set &R0, double t0, double t_final,
         const std::optional<Zonotope> &U, const Zonotope &U_state, double dt,
         const alg::HybridStandard &alg, const std::vector<Specification> &specs) {
    LocationReachSet R;
    R.location = index;
    R.time_point.push_back(R0);
    R.time_point_times.push_back({t0, t0});
    // CORA: tStart = tFinal or an empty invariant leave the start set alone
    if (within_tol(t0, t_final, numerics::kTimeRound) || priv_region_empty(loc.invariant()))
        return {R, true};
    const int steps = static_cast<int>(std::floor((t_final - t0) / dt + numerics::kStepCount));
    BundleSteps step(loc, R0, U, U_state, dt, alg);
    // without the time-point swap the analysis is CORALean's, its invariant test too
    const auto *inv = std::get_if<Polytope>(&loc.invariant());
    const bool rows = inv != nullptr && !alg.time_point_swap;
    for (int k = 0; k < steps; ++k) {
        // one step: the enclosure of [t_k, t_k+1], then the set at t_k+1
        const Set ti = step.next();
        R.time_interval.push_back(ti);
        R.time_interval_times.push_back({t0 + k * dt, t0 + (k + 1) * dt});
        R.time_point.push_back(step.time_point());
        R.time_point_times.push_back({t0 + (k + 1) * dt, t0 + (k + 1) * dt});
        if (!aux_specs_hold(specs, ti)) return {R, false};
        // the set has left the invariant (CORALean: a row of it lies strictly beyond the set)
        if (rows ? !priv_may_meet(inv->constraints(), ti) : !priv_meets(loc.invariant(), ti)) break;
    }
    return {R, true};
}

// The specifications ------------------------------------------------------------------------------

/// The image of the guard intersection under the reset. CORALean's W Rg + {w} adds the point {w}
/// as a zonotope with a zero generator per state; `lean` takes that for a zonotope and a reset
/// without input.
Set aux_reset(const LinearReset &reset, const Set &S, const std::optional<Zonotope> &U, bool lean) {
    if (!lean || !S.is<Zonotope>() || reset.B()) return reset.evaluate(S, U);
    const Zonotope image = S.as<Zonotope>().linear_map(reset.A());
    const Tensor w = reset.c() ? *reset.c() : image.center().zeros_like();
    const int64_t n = w.shape()[0];
    return image.plus(Zonotope(w, w.zeros_like({n, n})));
}

/// Whether a set may meet the invariant: the generic test of the pair for a polytope, the box
/// evaluation for a level set.
bool aux_meets_invariant(const Location &loc, const Set &S) {
    if (const auto *P = std::get_if<Polytope>(&loc.invariant())) return is_intersecting(*P, S);
    return std::get<LevelSet>(loc.invariant()).is_intersecting(*S);
}

/// The specifications that apply: an unsafe set only if it meets the invariant (CORA).
std::vector<Specification> aux_active(const Location &loc,
                                      const std::vector<Specification> &specs) {
    std::vector<Specification> out;
    for (const Specification &spec : specs) {
        if (spec.type() == SpecType::unsafe_set && !aux_meets_invariant(loc, spec.set())) continue;
        out.push_back(spec);
    }
    return out;
}

} // namespace


// ===========================================  MAIN  =========================================== //

Zonotope priv_state_input(const Location &loc, const std::optional<Zonotope> &U) {
    const int64_t n = loc.dim();
    // a nonlinear flow takes its input set itself, f(x, u) with u in U (none: the input 0)
    if (std::holds_alternative<NonlinearSys>(loc.flow())) {
        const int64_t m = loc.input_dim();
        if (U) return *U;
        return {loc.c().zeros_like({m, 1}), loc.c().zeros_like({m, 0})};
    }
    if (!U || !loc.B()) return {loc.c(), loc.c().zeros_like({n, 0})};
    const Tensor &B = *loc.B();
    // B U, then the constant input c, as CORALean's inputSetFloat
    return U->linear_map(B) + loc.c();
}

// The reach ---------------------------------------------------------------------------------------

LocationResult priv_location_reach(const Location &loc, int64_t index, const Set &R0,
                                   const TimeRange &t_start, double t_final,
                                   const std::optional<Zonotope> &U, double time_step,
                                   const alg::HybridStandard &alg,
                                   const std::vector<Specification> &specs,
                                   const std::vector<int64_t> &final_locations) {
    const Zonotope U_state = priv_state_input(loc, U);
    auto [R, verified] = aux_flow(loc, index, R0, t_start[0], t_final, U, U_state, time_step, alg,
                                  aux_active(loc, specs));
    LocationResult out{std::move(R), {}, verified};
    if (!verified) return out; // the analysis ends at a violation

    // the guards the sets meet, their intersections and resets (CORALean's boxes: dense)
    const bool lean = !alg.time_point_swap;
    const GuardHits hits = priv_pot_int(loc, out.R, final_locations, alg.time_point_swap);
    const double widen = t_start[1] - t_start[0];
    for (const GuardIntersection &g : priv_guard_intersect(
             loc, hits, out.R, U_state, alg.guard_intersect, alg.check_flow, lean)) {
        const Transition &trans = loc.transitions()[std::size_t(g.transition)];
        const std::vector<TimeRange> &times =
            hits.time_point ? out.R.time_point_times : out.R.time_interval_times;
        const TimeRange time{times[std::size_t(g.first)][0],
                             times[std::size_t(g.last)][1] + (hits.time_point ? 0 : widen)};
        // CORA turns the zonotope of a guard intersection back into a polynomial zonotope
        const Set before = alg.polynomialization && g.set.is<Zonotope>()
                               ? Set(PolyZonotope(g.set.as<Zonotope>()))
                               : g.set;
        out.jumps.push_back({aux_reset(trans.reset(), before, U, lean), trans.target(), time});
    }
    // an uncertain start time widens the times of the sets (CORA's updateTime)
    for (TimeRange &t : out.R.time_point_times) t[1] += widen;
    for (TimeRange &t : out.R.time_interval_times) t[1] += widen;
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
