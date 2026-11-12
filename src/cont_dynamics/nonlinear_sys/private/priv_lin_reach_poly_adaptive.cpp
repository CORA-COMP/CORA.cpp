// priv_lin_reach_poly_adaptive - one step of poly-adaptive: horizon, step size and gain
//
// As lin-adaptive (priv_lin_reach_adaptive), on polynomial zonotopes, with no tensor order to tune
// (it is 3). A step is passes at step sizes (priv_lin_reach_poly_adaptive_pass) in CORA's outer
// loop: the first run is at the finite horizon (the step size of the step before, scaled by the
// gain), saving its pass; its errors tune the step size (aux_optimaldeltat with dHused = 0.3), at
// which the second run takes the sets. Where the step size equals the horizon the horizon's pass is
// taken. The first step decreases its step size by the decrease factor until the gain is known,
// and the last one ends at t_final. After a step the zero-width dimensions, the orders of the error
// and the gain (aux_varphiest) are updated; with an empty third-order tensor the orders lose one at
// the second step (polyGain). A pass that fails halves the step size. CORALean's
// linReachPolyAdaptiveFloat, whose locals the structure Loop holds.
//
// Syntax:   PolyAdaptiveState next = priv_lin_reach_poly_adaptive(sys, U, alg, empty, t_final, st);
// Inputs:   see the declarations in priv_poly_adaptive.h
// Outputs:  next - the state after the step, its sets unreduced
// See also: priv_lin_reach_poly_adaptive_pass.cpp, priv_tune_adaptive.cpp
//
// CORA: contDynamics/@nonlinearSys/linReach_adaptive.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys/private/priv_adaptive_step.h"
#include "cont_dynamics/nonlinear_sys/private/priv_poly_adaptive.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The tuning of poly-adaptive: the constants and the pass; with an empty third-order tensor the
/// orders lose one at the second step (polyGain).
struct Tune {
    const NonlinearSys &sys;
    const std::optional<Zonotope> &U;
    const alg::PolynomializationAdaptive &alg;
    bool third_order_empty;
    double t_final;
    double decr_factor, zeta_z;
    std::vector<double> zeta_phi;
    std::optional<double> d_h_used;
    int fuel;

    PolyAdaptiveState start(const PolyAdaptiveState &st, bool last) const {
        PolyAdaptiveState out = st;
        if (!(st.i == 2 && third_order_empty && !last)) return out;
        // the orders lose one (to 0 at most), and the least of them is found again
        int minorder = -1;
        for (int &q : out.orders) {
            if (q >= 0) q = std::max(q - 1, 0);
            if (q >= 0) minorder = minorder < 0 ? q : std::min(minorder, q);
        }
        out.minorder = std::max(minorder, 0);
        return out;
    }
    std::optional<PolyAdaptivePass> run_pass(const PolyAdaptiveState &st, double time_step,
                                             const Vec &err_adm) const {
        return priv_lin_reach_poly_adaptive_pass(sys, U, alg, time_step, st.R, err_adm);
    }
    PolyAdaptiveState update_orders(const PolyAdaptiveState &st, bool veryfirststep,
                                    const PolyAdaptivePass &P) const {
        // the orders of the error terms from the pattern of A and the Hessian at the point
        PolyAdaptiveState out = st;
        OrdersState o{st.zero_width, st.orders, st.minorder};
        priv_update_orders(sys, U, st.R.center(), priv_row_sums(st.R.zonotope()), veryfirststep,
                           P.A, P.xs, P.gzero, third_order_empty ? 1 : 0, o);
        out.zero_width = o.zero_width;
        out.orders = o.orders;
        out.minorder = o.minorder;
        return out;
    }
    PolyAdaptiveState after_run(PolyAdaptiveState st, const PolyAdaptivePass &, double,
                                bool) const {
        return st;
    }
    double radius(const PolyAdaptiveState &st) const {
        return priv_norm_up(priv_poly_radius(st.R));
    }
    std::vector<char> est_mask(const std::vector<char> &mask) const {
        return std::vector<char>(mask.size(), 1); // aux_varphiest runs over all dimensions
    }
};

} // namespace


// ===========================================  MAIN  =========================================== //

// One step ----------------------------------------------------------------------------------------

PolyAdaptiveState priv_lin_reach_poly_adaptive(const NonlinearSys &sys,
                                               const std::optional<Zonotope> &U,
                                               const alg::PolynomializationAdaptive &alg,
                                               bool third_order_empty, double t_final,
                                               const PolyAdaptiveState &st) {
    const Tune tune{sys,          U,
                    alg,          third_order_empty,
                    t_final,      alg.decr_factor,
                    alg.zeta_z,   alg.zeta_phi,
                    alg.d_h_used, alg.fuel};
    return priv_adaptive_step<PolyAdaptiveState, PolyAdaptivePass>(tune, st);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
