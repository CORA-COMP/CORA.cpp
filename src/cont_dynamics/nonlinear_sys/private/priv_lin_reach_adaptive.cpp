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

#include "cont_dynamics/nonlinear_sys/private/priv_adaptive_step.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The tuning of lin-adaptive: the constants, the pass at a tensor order and the tensor order.
struct Tune {
    const NonlinearSys &sys;
    const std::optional<Zonotope> &U;
    const alg::LinearizationAdaptive &alg;
    double t_final;
    int64_t n, m;
    double decr_factor, zeta_z;
    std::vector<double> zeta_phi;
    std::optional<double> d_h_used;
    int fuel;

    AdaptiveState start(const AdaptiveState &st, bool last) const;
    std::optional<AdaptivePass> run_pass(const AdaptiveState &st, double time_step,
                                         const Vec &err_adm) const;
    AdaptiveState update_orders(const AdaptiveState &st, bool veryfirststep,
                                const AdaptivePass &P) const;
    AdaptiveState after_run(AdaptiveState st, const AdaptivePass &P, double time_step,
                            bool last_step) const;
    double radius(const AdaptiveState &st) const { return priv_norm_up(priv_row_sums(st.R)); }
    std::vector<char> est_mask(const std::vector<char> &mask) const { return mask; }
};

/// The expansion matrices B at (xs, us).
Tensor aux_B(const Tune &c, const Tensor &like, const Vec &xs, const Vec &us) {
    return c.sys
        .jacobian(priv_column(like, xs),
                  c.m > 0 ? std::optional<Tensor>(priv_column(like, us)) : std::nullopt)
        .B;
}

/// The tensor order of the first step: 2 from a point, else 2 unless the third-order error is
/// above zeta_k times the second-order one in every dimension that has one
/// (aux_initStepTensorOrder).
int aux_init_tensor_order(const Tune &c, const Zonotope &R) {
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

/// The tensor order of the next step: when the predicted error moved by more than 1 - zeta_k, the
/// other order's error at the same Rmax, and the switch if it wins by zeta_k.
AdaptiveState aux_next_tensor_order(const Tune &c, const AdaptiveState &st, const AdaptivePass &P,
                                    double time_step) {
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

} // namespace


// ===========================================  MAIN  =========================================== //

// The tuning --------------------------------------------------------------------------------------

AdaptiveState Tune::start(const AdaptiveState &st, bool) const {
    // the very first step chooses the tensor order from the two errors
    AdaptiveState out = st;
    if (st.i == 1) out.tensor_order = aux_init_tensor_order(*this, st.R);
    return out;
}

std::optional<AdaptivePass> Tune::run_pass(const AdaptiveState &st, double time_step,
                                           const Vec &err_adm) const {
    return priv_lin_reach_adaptive_pass(sys, st.tensor_order, U, alg, time_step, st.R, err_adm);
}

AdaptiveState Tune::update_orders(const AdaptiveState &st, bool veryfirststep,
                                  const AdaptivePass &P) const {
    // the orders of the error terms from the pattern of A and the Hessian at the point
    AdaptiveState out = st;
    OrdersState o{st.zero_width, st.orders, st.minorder};
    priv_update_orders(sys, U, st.R.center(), priv_row_sums(st.R), veryfirststep, P.A, P.xs,
                       P.gzero, 0, o);
    out.zero_width = o.zero_width;
    out.orders = o.orders;
    out.minorder = o.minorder;
    return out;
}

AdaptiveState Tune::after_run(AdaptiveState st, const AdaptivePass &P, double time_step,
                              bool last_step) const {
    // the first step records the errors of its step size, a later one may switch the tensor order
    if (st.i == 1) {
        st.kappa_deltat = time_step;
        st.kappa_abstrerr = priv_norm_up(P.abstrerr);
    } else if (!last_step) {
        st = aux_next_tensor_order(*this, st, P, time_step);
    }
    return st;
}

// One step ----------------------------------------------------------------------------------------

AdaptiveState priv_lin_reach_adaptive(const NonlinearSys &sys, const std::optional<Zonotope> &U,
                                      const alg::LinearizationAdaptive &alg, double t_final,
                                      const AdaptiveState &st) {
    const Tune tune{sys,
                    U,
                    alg,
                    t_final,
                    sys.dim(),
                    sys.input_dim(),
                    alg.decr_factor,
                    alg.zeta_z,
                    alg.zeta_phi,
                    std::nullopt,
                    alg.fuel};
    return priv_adaptive_step<AdaptiveState, AdaptivePass>(tune, st);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
