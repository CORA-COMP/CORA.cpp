// priv_adapt_step - one adaptively tuned step of a nonlinear system (Alg. 2 of Wetzlinger et al.)
//
// The finite horizon h_k follows from the gain of the step before (eq. (59)). A first pass at the
// horizon measures the radius of the abstraction error, from which the step size is searched
// (eq. (58)); a second pass at that step size gives the sets, the radius and the diameters of the
// error at both tensor orders, which choose the order of the next step. The gain of the step is
// the one that explains the two radii (eq. (56)). The sets are reduced to the zonotope order.
//
// Syntax:   AdaptState next = priv_adapt_step(sys, U, alg, state);
// Inputs:   sys, U - the system and the input set (none: no inputs), alg, state - the last step
// Outputs:  next - the state after the step
// References:
//    [1] M. Wetzlinger, A. Kulmburg, A. Le Penven and M. Althoff. "Adaptive Reachability
//        Algorithms for Nonlinear Systems Using Abstraction Error Analysis", NAHS, 2022.
// See also: priv_order_step.cpp, priv_tune_adaptive.cpp
//
// CORA: contDynamics/@nonlinearSys/linReach_adaptive.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       10-November-2026
// Last update:   10-November-2026
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys/private/priv_adaptive.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The sizes of the sets ---------------------------------------------------------------------------

/// The largest radius of the box of Z.
double aux_rad(const Zonotope &Z) {
    const Interval I = Z.interval();
    const Vec lo = I.inf().data(), hi = I.sup().data();
    double r = 0;
    for (std::size_t i = 0; i < lo.size(); ++i) r = std::max(r, 0.5 * (hi[i] - lo[i]));
    return r;
}

/// The diameter of the box of Z per coordinate.
Vec aux_dia(const Zonotope &Z) {
    const Interval I = Z.interval();
    const Vec lo = I.inf().data(), hi = I.sup().data();
    Vec d(lo.size());
    for (std::size_t i = 0; i < lo.size(); ++i) d[i] = hi[i] - lo[i];
    return d;
}

/// max(0, dia(F) - dia(P)) per coordinate: the diameter the abstraction error adds.
Vec aux_added(const Zonotope &F, const Zonotope &P) {
    Vec f = aux_dia(F);
    const Vec p = aux_dia(P);
    for (std::size_t i = 0; i < f.size(); ++i) f[i] = std::max(0.0, f[i] - p[i]);
    return f;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The step ----------------------------------------------------------------------------------------

AdaptState priv_adapt_step(const NonlinearSys &sys, const std::optional<Zonotope> &U,
                           const alg::LinearizationAdaptive &alg, const AdaptState &s) {
    const double zd = alg.zeta_delta;
    // 1. the finite horizon h_k, eq. (59), changing by at most the factor zeta_delta per step
    double hk = std::clamp(s.h * (zd - alg.zeta_h) / (zd - s.phi), s.h * zd, s.h / zd);
    if (!(hk > 0 && hk < 1e300))
        CORA_FAIL(Error::reach_set_explosion, "reach: the finite horizon " + std::to_string(hk) +
                                                  " is not positive (the gain is " +
                                                  std::to_string(s.phi) + ")");
    // 3.-9. the passes at the horizon, which is scaled down while the flow leaves the region
    std::optional<AdaptPass> pass1;
    std::optional<LinPass> p1;
    std::optional<StepSets> F1;
    for (int k = 0; !F1; ++k) {
        try {
            pass1.emplace(sys, U, alg, hk, s.R);
            p1 = pass1->lin_pass();
            F1 = pass1->order_step(s.kappa);
        } catch (const Error &e) {
            if (k >= alg.fuel) CORA_FAIL(e.id(), e.what());
            hk = hk * zd;
        }
    }
    const double rabs1 = std::max(0.0, aux_rad(F1->time_point) - aux_rad(p1->sets.time_point));
    // 10. the step size, with e^{tr(A) h} of eq. (50)
    const double trace = p1->A.diagonal().sum().item();
    const double zeta_a = priv_exp_up(alg.exp_depth, trace * hk);
    const StepSize opt = priv_opt_delta_t(alg, zeta_a, s.phi, aux_rad(s.R), rabs1, hk);
    if (!(opt.time_step > numerics::kMinTimeStep))
        CORA_FAIL(Error::not_converged,
                  "reach: the step size converges to 0 (" + std::to_string(opt.time_step) + ")");
    // 3.-12. the passes at the step size; at the horizon itself they are those above
    const bool same = opt.exponent == 0;
    std::optional<AdaptPass> pass2;
    if (!same) pass2.emplace(sys, U, alg, opt.time_step, s.R);
    AdaptPass &pass = same ? *pass1 : *pass2;
    const LinPass p2 = same ? *p1 : pass.lin_pass();
    const StepSets G1 = same && s.kappa == 1 ? *F1 : pass.order_step(1);
    const StepSets G2 = same && s.kappa == 2 ? *F1 : pass.order_step(2);
    // 12. the diameters of the error at both orders, and the radius at the order kappa
    const Vec dabs1 = aux_added(G1.time_point, p2.sets.time_point);
    const Vec dabs2 = aux_added(G2.time_point, p2.sets.time_point);
    const StepSets &F = s.kappa == 1 ? G1 : G2;
    const double rabs_opt = std::max(0.0, aux_rad(F.time_point) - aux_rad(p2.sets.time_point));
    // 13. the order of the next step, 16. the gain
    const int kappa = priv_tune_order(alg.zeta_k, s.phi, rabs_opt, s.rabs, dabs1, dabs2, s.kappa);
    const double lam = priv_pow(1 / zd, opt.exponent);
    const double phi = priv_estimate_gain(zd, lam, rabs1, rabs_opt, opt.exponent, alg.fuel);
    // 17.-18. the time and the reduced sets
    return {s.t + opt.time_step,
            F.time_point.reduce(alg.zonotope_order),
            F.time_interval.reduce(alg.zonotope_order),
            hk,
            phi,
            kappa,
            rabs_opt};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
