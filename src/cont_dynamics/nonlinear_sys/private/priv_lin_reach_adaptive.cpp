// priv_lin_reach_adaptive - one step of lin-adaptive: horizon, step size, gain and tensor order
//
// A step is passes at step sizes (priv_lin_reach_adaptive_pass) in CORA's outer loop. The first
// run is at the finite horizon (the step size of the step before, scaled by the gain), saving its
// pass; its errors tune the step size (aux_optimaldeltat), at which the second run takes the sets.
// Where the step size equals the horizon the horizon's pass is taken. The first step decreases its
// step size by the decrease factor until the gain is known, and the last one ends at t_final.
// After a step the zero-width dimensions, the orders of the error, the gain (aux_varphiest) and
// the tensor order are updated. A pass that fails halves the step size. CORALean's
// linReachAdaptiveFloat, whose locals the structure Loop holds.
//
// Syntax:   AdaptiveState next = priv_lin_reach_adaptive(sys, U, alg, t_final, state);
// Inputs:   see the declarations in priv_adaptive.h
// Outputs:  next - the state after the step, its sets unreduced
// See also: priv_lin_reach_adaptive_pass.cpp, priv_tune_adaptive.cpp
//
// CORA: contDynamics/@nonlinearSys/linReach_adaptive.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys/private/priv_adaptive.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The locals of the outer loop of a step (CORALean's LinReachAdaptiveLoop).
struct Loop {
    AdaptiveState st;
    double time_step, finitehorizon;
    Vec err_adm;
    int run;
    bool last_step, veryfirststep, equal_horizon;
    double zeta_p, zeta_p_h;
    std::optional<std::pair<double, AdaptivePass>> horizon; ///< the pass saved at the horizon
};

/// The context of a step.
struct Context {
    const NonlinearSys &sys;
    const std::optional<Zonotope> &U;
    const alg::LinearizationAdaptive &alg;
    double t_final;
    int64_t n, m;
};

// Helpers -----------------------------------------------------------------------------------------

/// withinTol(a, b, tol): absolutely or relatively within tol.
bool aux_within_tol(double a, double b, double tol) {
    const double d = std::abs(a - b);
    return d <= tol || d / std::min(std::abs(a), std::abs(b)) <= tol;
}

/// max(errRatio(~isnan(errRatio))) over the dimensions of the mask, 0/0 skipped.
double aux_max_ratio(const std::vector<char> &mask, const Vec &num, const Vec &den) {
    double best = 0;
    for (std::size_t i = 0; i < num.size(); ++i) {
        const double r = mask[i] && !(num[i] <= 0 && den[i] <= 0) ? num[i] / den[i] : 0.0;
        best = i == 0 ? r : std::max(best, r);
    }
    return best;
}

/// The expansion matrices B at (xs, us).
Tensor aux_B(const Context &c, const Tensor &like, const Vec &xs, const Vec &us) {
    return c.sys
        .jacobian(priv_column(like, xs),
                  c.m > 0 ? std::optional<Tensor>(priv_column(like, us)) : std::nullopt)
        .B;
}

/// The tensor order of the first step: 2 from a point, else 2 unless the third-order error is
/// above zeta_k times the second-order one in every dimension that has one
/// (aux_initStepTensorOrder).
int aux_init_tensor_order(const Context &c, const Zonotope &R) {
    if (R.generators().shape().back() == 0) return 2;
    const Vec xs = R.center().data(), us = c.m > 0 ? c.U->center().data() : Vec{};
    const Tensor &like = R.center();
    const Jacobians J =
        c.sys.jacobian(priv_column(like, xs),
                       c.m > 0 ? std::optional<Tensor>(priv_column(like, us)) : std::nullopt);
    const Vec L2 = priv_norm_mag(
        priv_zonotope_box(priv_abstr_err_adaptive(c.sys, 2, J.A, J.B, xs, R, c.U, c.alg.zeta_z)));
    const Vec L3 = priv_norm_mag(
        priv_zonotope_box(priv_abstr_err_adaptive(c.sys, 3, J.A, J.B, xs, R, c.U, c.alg.zeta_z)));
    for (std::size_t i = 0; i < L2.size(); ++i)
        if (L2[i] <= 0 || !(L3[i] / L2[i] <= c.alg.zeta_k)) return 2;
    return 3;
}

// The orders --------------------------------------------------------------------------------------

/// The orders of the error and the zero-width dimensions after a pass (aux_getPowers), updated at
/// the very first step or when a zero-width dimension changes.
AdaptiveState aux_orders(const Context &c, const Loop &L, const AdaptivePass &P) {
    AdaptiveState st = L.st;
    // a dimension has zero width where its generators sum to almost nothing
    const Vec row = priv_row_sums(st.R);
    std::vector<char> zw(row.size());
    bool changed = false;
    for (std::size_t i = 0; i < zw.size(); ++i) {
        zw[i] = row[i] <= numerics::kZeroWidth;
        changed = changed || zw[i] != st.zero_width[i];
    }
    if (!(L.veryfirststep || changed)) return st;
    // the Hessian at the linearization point gives the orders of the error terms
    const Tensor &like = st.R.center();
    const Vec us = c.m > 0 ? c.U->center().data() : Vec{};
    const auto H = c.sys.hessian_enclosure(
        Range(priv_column(like, P.xs)),
        c.m > 0 ? std::optional<Range>(Range(priv_column(like, us))) : std::nullopt);
    st.zero_width = zw;
    st.orders = priv_adaptive_orders(P.A, zw, P.gzero, H, c.n, c.m, 100);
    // the least order of the dimensions with an error
    int minorder = -1;
    for (const int q : st.orders)
        if (q >= 0) minorder = minorder < 0 ? q : std::min(minorder, q);
    st.minorder = std::max(minorder, 0);
    return st;
}

/// zetaP: exp(trace(A dt)) at the first run, else as it was.
double aux_zeta_p(const Loop &L, const AdaptivePass &P) {
    if (L.run != 1) return L.zeta_p;
    const Vec a = P.A.data();
    const int64_t n = int64_t(P.xs.size());
    Vec diag(static_cast<std::size_t>(n));
    for (int64_t i = 0; i < n; ++i) diag[std::size_t(i)] = a[std::size_t(i * n + i)] * L.time_step;
    return std::exp(priv_sum_up(diag));
}

/// The mask of the dimensions of the minimal order.
std::vector<char> aux_mask(const AdaptiveState &st) {
    std::vector<char> mask(st.orders.size());
    for (std::size_t i = 0; i < mask.size(); ++i) mask[i] = st.orders[i] == st.minorder;
    return mask;
}

// The tensor order --------------------------------------------------------------------------------

/// The tensor order of the next step: when the predicted error moved by more than 1 - zeta_k, the
/// other order's error at the same Rmax, and the switch if it wins by zeta_k.
AdaptiveState aux_next_tensor_order(const Context &c, const AdaptiveState &st,
                                    const AdaptivePass &P, double time_step) {
    // the error the gain predicts at this step size, against the one measured
    const double lim = priv_pow(c.alg.decr_factor, st.minorder + 1);
    const double total =
        priv_varphi_total(st.varphi, lim, st.kappa_deltat, time_step, c.alg.decr_factor);
    const double abstrerr = priv_norm_up(P.abstrerr);
    if (std::abs((total * abstrerr) / st.kappa_abstrerr - 1) <= 1 - c.alg.zeta_k) return st;
    // the other order's error at the same Rmax, through the linear step
    const Tensor &like = st.R.center();
    const int other_order = st.tensor_order == 2 ? 3 : 2;
    const Vec us = c.m > 0 ? c.U->center().data() : Vec{};
    const Tensor B = aux_B(c, like, P.xs, us);
    const Zonotope V =
        priv_abstr_err_adaptive(c.sys, other_order, P.A, B, P.xs, P.Rmax, c.U, c.alg.zeta_z);
    const Zonotope Rerror =
        LinearSys(P.A).error_solution_adaptive(V, std::nullopt, time_step, 0.0, c.alg.taylor_terms);
    const double other = priv_norm_up(priv_row_sums(Rerror));
    const double ratio = st.tensor_order == 2 ? other / abstrerr : abstrerr / other;
    const bool switches = st.tensor_order == 2 ? ratio <= c.alg.zeta_k : !(ratio <= c.alg.zeta_k);
    if (!switches) return st;
    // the switch: the errors of the new order start the tuning again
    AdaptiveState out = st;
    out.tensor_order = other_order;
    out.kappa_deltat = time_step;
    out.kappa_abstrerr = other;
    out.err_adm_deltatopt = priv_norm_mag(priv_zonotope_box(V));
    out.err_adm_horizon = Vec(st.err_adm_horizon.size(), 0.0);
    return out;
}

// The runs ----------------------------------------------------------------------------------------

/// The first run, at the finite horizon, after a pass: the locals the loop continues with (run 0
/// where CORA's continue skips the counter).
Loop aux_horizon(const Context &c, const Loop &L, const AdaptivePass &P) {
    const double decr = c.alg.decr_factor;
    Loop out = L;
    const double zeta_p = aux_zeta_p(L, P);
    const AdaptiveState st = aux_orders(c, L, P);
    const std::vector<char> mask = aux_mask(st);
    const double zetaphi = c.alg.zeta_phi[std::size_t(std::min(st.minorder, 2))];
    const AdaptivePass &hor = L.horizon ? L.horizon->second : P;
    const double r_radius = priv_norm_up(priv_row_sums(st.R));
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
        const double varphi = aux_max_ratio(mask, P.abstrerr, hor.abstrerr);
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
        out.time_step =
            priv_optimal_delta_t(decr, c.alg.zeta_z, r_radius, priv_norm_up(hor.abstrerr),
                                 L.finitehorizon, varphi, L.zeta_p_h);
        out.err_adm = Vec(L.err_adm.size(), 0.0);
        return out;
    }
    // a later step: the step size from the error of this pass, not above the time left
    const double dt = priv_optimal_delta_t(decr, c.alg.zeta_z, r_radius, priv_norm_up(P.abstrerr),
                                           L.finitehorizon, st.varphi, zeta_p);
    out.st = st;
    out.st.err_adm_horizon = P.true_error;
    out.zeta_p = zeta_p;
    out.horizon = std::make_pair(L.time_step, P);
    out.err_adm = st.err_adm_deltatopt;
    out.time_step = dt <= c.t_final - st.t ? dt : c.t_final - st.t;
    return out;
}

/// The second run (at the tuned step size), or the last step's only one: the locals the loop ends
/// on, with the gain, the errors and the tensor order of the next step.
Loop aux_finish(const Context &c, const Loop &L, const AdaptivePass &P) {
    // the gain of this step: the ratio, or the estimate for the step size that was tuned
    AdaptiveState st = aux_orders(c, L, P);
    const std::vector<char> mask = aux_mask(st);
    // the horizon's pass, or this one where there is none
    const AdaptivePass &hor = L.horizon ? L.horizon->second : P;
    double varphi = st.varphi;
    if (L.equal_horizon)
        varphi = aux_max_ratio(mask, P.abstrerr, hor.abstrerr);
    else if (!L.last_step)
        varphi = priv_varphi_est(L.finitehorizon, L.time_step, hor.abstrerr, P.abstrerr,
                                 c.alg.decr_factor, mask, st.minorder, c.alg.fuel);
    // the gain, the horizon and the error of this step go to the next one
    st.varphi = varphi;
    st.finitehorizon = L.finitehorizon;
    st.err_adm_deltatopt = P.true_error;
    if (st.i == 1) {
        st.kappa_deltat = L.time_step;
        st.kappa_abstrerr = priv_norm_up(P.abstrerr);
    } else if (!L.last_step) {
        st = aux_next_tensor_order(c, st, P, L.time_step);
    }
    Loop out = L;
    out.st = st;
    out.zeta_p = aux_zeta_p(L, P);
    return out;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// One step ----------------------------------------------------------------------------------------

AdaptiveState priv_lin_reach_adaptive(const NonlinearSys &sys, const std::optional<Zonotope> &U,
                                      const alg::LinearizationAdaptive &alg, double t_final,
                                      const AdaptiveState &st) {
    const Context c{sys, U, alg, t_final, sys.dim(), sys.input_dim()};
    // the finite horizon from the gain of the step before, and the step size of the first pass
    const double rem = t_final - st.t;
    const double zetaphi = alg.zeta_phi[std::size_t(std::min(st.minorder, 2))];
    const double fh = st.finitehorizon * ((1 + st.varphi) - zetaphi);
    double h0 = fh;
    bool last = false, first = false;
    if (st.i == 1) {
        h0 = t_final * 0.01;
        first = true;
    } else if (rem < st.time_step) {
        h0 = rem;
        last = true;
    } else if (aux_within_tol(fh, st.time_step, numerics::kHorizonMatch)) {
        h0 = st.finitehorizon * 1.1;
    }
    // the very first step chooses the tensor order from the two errors
    AdaptiveState st0 = st;
    if (st.i == 1) st0.tensor_order = aux_init_tensor_order(c, st.R);

    // CORA's while true: a pass at the step size, halved where it fails
    Loop L{st0, h0, h0, st.err_adm_horizon, 1, last, first, false, 0.0, 0.0, std::nullopt};
    std::optional<AdaptivePass> final_pass;
    for (int fuel = 0; fuel < alg.fuel && !final_pass; ++fuel) {
        const auto P = priv_lin_reach_adaptive_pass(sys, L.st.tensor_order, U, alg, L.time_step,
                                                    L.st.R, L.err_adm);
        if (!P) {
            // ~Lconverged: halve
            L.finitehorizon = 0.5 * L.time_step;
            L.time_step = 0.5 * L.time_step;
            L.err_adm = L.st.err_adm_horizon;
            continue;
        }
        // the first run tunes the step size, the second (or the only) one ends the step
        if (L.run == 1 && !L.last_step) {
            Loop next = aux_horizon(c, L, *P);
            if (next.run == 0) {
                next.run = 1;
            } else if (next.time_step <= next.finitehorizon &&
                       next.finitehorizon <= next.time_step) {
                next.run = next.run + 1;
                next.equal_horizon = true;
                next.time_step = next.time_step * alg.decr_factor;
            } else {
                next.run = next.run + 1;
            }
            L = std::move(next);
        } else {
            L = aux_finish(c, L, *P);
            final_pass = *P;
        }
    }
    // a loop that ran out of fuel is no step
    if (!final_pass)
        CORA_FAIL(Error::not_converged,
                  "reach: the step of the adaptive algorithm did not converge");

    // the sets of the step: those of the horizon where the step size equals it
    const std::pair<double, AdaptivePass> F =
        L.equal_horizon && L.horizon ? *L.horizon : std::make_pair(L.time_step, *final_pass);
    AdaptiveState next = L.st;
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
