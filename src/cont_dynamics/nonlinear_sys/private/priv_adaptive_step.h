// priv_adaptive_step - the step of lin-adaptive and poly-adaptive, one template
//
// One step of CORA's linReach_adaptive: passes at step sizes in CORA's outer loop. The first run
// is at the finite horizon (the step size of the step before, scaled by the gain), saving its pass;
// its errors tune the step size (aux_optimaldeltat), at which the second run takes the sets. Where
// the step size equals the horizon the horizon's pass is taken. The first step decreases its step
// size by the decrease factor until the gain is known, and the last one ends at t_final. After a
// step the zero-width dimensions, the orders of the error and the gain (aux_varphiest) are
// updated. A pass that fails halves the step size. CORALean's linReachAdaptiveFloat and
// linReachPolyAdaptiveFloat, whose locals the structure StepLoop holds. The algorithms differ in
// the state, the pass and the Tune (the constants, the pass, the orders, and what happens after a
// step), which the template takes.
//
// Syntax:   State next = priv_adaptive_step<State, Pass>(tune, state);
// Inputs:   tune - the constants of the algorithm and its pass, orders and tensor order
//           state - the state before the step
// Outputs:  next - the state after the step, its sets unreduced
// See also: priv_lin_reach_adaptive.cpp, priv_lin_reach_poly_adaptive.cpp
//
// CORA: contDynamics/@nonlinearSys/linReach_adaptive.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cont_dynamics/nonlinear_sys/private/priv_adaptive.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// The locals --------------------------------------------------------------------------------------

/// The locals of the outer loop of a step (CORALean's LinReachAdaptiveLoop).
template <class State, class Pass>
struct StepLoop {
    State st;
    double time_step, finitehorizon;
    Vec err_adm;
    int run;
    bool last_step, veryfirststep, equal_horizon;
    double zeta_p, zeta_p_h;
    std::optional<std::pair<double, Pass>> horizon; ///< the pass saved at the horizon
};

/// The mask of the dimensions of the minimal order.
inline std::vector<char> priv_order_mask(const std::vector<int> &orders, int minorder) {
    std::vector<char> mask(orders.size());
    for (std::size_t i = 0; i < mask.size(); ++i) mask[i] = orders[i] == minorder;
    return mask;
}

// The runs ----------------------------------------------------------------------------------------

/// The first run, at the finite horizon, after a pass: the locals the loop continues with (run 0
/// where CORA's continue skips the counter).
template <class State, class Pass, class Tune>
StepLoop<State, Pass> priv_horizon_run(const Tune &c, const StepLoop<State, Pass> &L,
                                       const Pass &P) {
    const double decr = c.decr_factor;
    StepLoop<State, Pass> out = L;
    const double zeta_p = L.run != 1 ? L.zeta_p : priv_zeta_p(P.A, L.time_step);
    const State st = c.update_orders(L.st, L.veryfirststep, P);
    const std::vector<char> mask = priv_order_mask(st.orders, st.minorder);
    const double zetaphi = c.zeta_phi[std::size_t(std::min(st.minorder, 2))];
    const Pass &hor = L.horizon ? L.horizon->second : P;
    const double r_radius = c.radius(st);
    Vec decr_err = P.true_error;
    for (double &e : decr_err) e = decr * e;
    if (st.i == 1) {
        if (L.veryfirststep) {
            // save the pass, decrease the step size, and again
            out.st = st;
            out.veryfirststep = false;
            out.zeta_p = out.zeta_p_h = zeta_p;
            out.horizon = std::make_pair(L.time_step, P);
            out.time_step = decr * L.time_step;
            out.err_adm = decr_err;
            out.run = 0;
            return out;
        }
        // the gain from the two passes: below zeta_phi the horizon decreases again
        const double varphi = priv_max_ratio(mask, P.abstrerr, hor.abstrerr);
        if (!(zetaphi <= varphi)) {
            out.st = st;
            out.zeta_p = out.zeta_p_h = zeta_p;
            out.horizon = std::make_pair(L.time_step, P);
            out.finitehorizon = L.time_step;
            out.time_step = decr * L.time_step;
            out.err_adm = decr_err;
            out.run = 0;
            return out;
        }
        // the gain is large enough: tune the step size of the second run
        out.st = st;
        out.st.varphi = varphi;
        out.st.finitehorizon = L.finitehorizon;
        out.st.err_adm_horizon = P.true_error;
        out.zeta_p = zeta_p;
        out.time_step = priv_optimal_delta_t(decr, c.zeta_z, r_radius, priv_norm_up(hor.abstrerr),
                                             L.finitehorizon, varphi, L.zeta_p_h, c.d_h_used);
        out.err_adm = Vec(L.err_adm.size(), 0.0);
        return out;
    }
    // a later step: the step size from the error of this pass, not above the time left
    const double dt = priv_optimal_delta_t(decr, c.zeta_z, r_radius, priv_norm_up(P.abstrerr),
                                           L.finitehorizon, st.varphi, zeta_p, c.d_h_used);
    out.st = st;
    out.st.err_adm_horizon = P.true_error;
    out.zeta_p = zeta_p;
    out.horizon = std::make_pair(L.time_step, P);
    out.err_adm = st.err_adm_deltatopt;
    out.time_step = dt <= c.t_final - st.t ? dt : c.t_final - st.t;
    return out;
}

/// The second run (at the tuned step size), or the last step's only one: the locals the loop ends
/// on, with the gain and the errors of the next step.
template <class State, class Pass, class Tune>
StepLoop<State, Pass> priv_finish_run(const Tune &c, const StepLoop<State, Pass> &L,
                                      const Pass &P) {
    // the gain of this step: the ratio, or the estimate for the step size that was tuned
    State st = c.update_orders(L.st, L.veryfirststep, P);
    const std::vector<char> mask = priv_order_mask(st.orders, st.minorder);
    // the horizon's pass, or this one where there is none
    const Pass &hor = L.horizon ? L.horizon->second : P;
    double varphi = st.varphi;
    if (L.equal_horizon)
        varphi = priv_max_ratio(mask, P.abstrerr, hor.abstrerr);
    else if (!L.last_step)
        varphi = priv_varphi_est(L.finitehorizon, L.time_step, hor.abstrerr, P.abstrerr,
                                 c.decr_factor, c.est_mask(mask), st.minorder, c.fuel);
    // the gain, the horizon and the error of this step go to the next one
    st.varphi = varphi;
    st.finitehorizon = L.finitehorizon;
    st.err_adm_deltatopt = P.true_error;
    st = c.after_run(st, P, L.time_step, L.last_step);
    StepLoop<State, Pass> out = L;
    out.st = st;
    out.zeta_p = L.run != 1 ? L.zeta_p : priv_zeta_p(P.A, L.time_step);
    return out;
}

// One step ----------------------------------------------------------------------------------------

template <class State, class Pass, class Tune>
State priv_adaptive_step(const Tune &c, const State &st) {
    const double t_final = c.t_final;
    // the finite horizon from the gain of the step before, and the step size of the first pass
    const double rem = t_final - st.t;
    const double zetaphi = c.zeta_phi[std::size_t(std::min(st.minorder, 2))];
    const double fh = st.finitehorizon * ((1 + st.varphi) - zetaphi);
    double h0 = fh;
    bool last = false, first = false;
    if (st.i == 1) {
        h0 = t_final * 0.01;
        first = true;
    } else if (rem < st.time_step) {
        h0 = rem;
        last = true;
    } else if (priv_within_tol(fh, st.time_step, numerics::kHorizonMatch)) {
        h0 = st.finitehorizon * 1.1;
    }
    // the state the step starts from (the tensor order of the first step, the gain of the second)
    const State st0 = c.start(st, last);

    // CORA's while true: a pass at the step size, halved where it fails
    StepLoop<State, Pass> L{st0,   h0,  h0,  st.err_adm_horizon, 1, last, first,
                            false, 0.0, 0.0, std::nullopt};
    std::optional<Pass> final_pass;
    for (int fuel = 0; fuel < c.fuel && !final_pass; ++fuel) {
        const auto P = c.run_pass(L.st, L.time_step, L.err_adm);
        if (!P) {
            // ~Lconverged: halve
            L.finitehorizon = 0.5 * L.time_step;
            L.time_step = 0.5 * L.time_step;
            L.err_adm = L.st.err_adm_horizon;
            continue;
        }
        // the first run tunes the step size, the second (or the only) one ends the step
        if (L.run == 1 && !L.last_step) {
            StepLoop<State, Pass> next = priv_horizon_run(c, L, *P);
            if (next.run == 0) {
                next.run = 1;
            } else if (next.time_step <= next.finitehorizon &&
                       next.finitehorizon <= next.time_step) {
                next.run = next.run + 1;
                next.equal_horizon = true;
                next.time_step = next.time_step * c.decr_factor;
            } else {
                next.run = next.run + 1;
            }
            L = std::move(next);
        } else {
            L = priv_finish_run(c, L, *P);
            final_pass = *P;
        }
    }
    // a loop that ran out of fuel is no step
    if (!final_pass)
        CORA_FAIL(Error::not_converged,
                  "reach: the step of the adaptive algorithm did not converge");

    // the sets of the step: those of the horizon where the step size equals it
    const std::pair<double, Pass> F =
        L.equal_horizon && L.horizon ? *L.horizon : std::make_pair(L.time_step, *final_pass);
    State next = L.st;
    next.i = st.i + 1;
    next.t = st.t + F.first;
    next.R = F.second.Rtp;
    next.Rti = F.second.Rti;
    next.time_step = F.first;
    next.steps.insert(next.steps.begin(), F.first);
    return next;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
