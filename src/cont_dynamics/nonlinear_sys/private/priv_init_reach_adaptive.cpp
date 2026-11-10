// priv_init_reach_adaptive - one step of the linearized system with the series cut adaptively
//
// The affine system x' = A x + u, u in U + u_const, from the set Rstart as CORA's linearSys
// initReach_adaptive solves it: the Taylor series of e^{A dt} ends where a further term gains less
// than zeta_t_lin of the Frobenius norm of the curvature matrix (or at the number the pass before
// used in this step), its remainder is an interval matrix E, and the sets are reduced adaptively.
// The result is the set over the step and at its end. If the series does not converge within 50
// terms the step is too large, and there is no result.
//
// Syntax:   auto R = priv_init_reach_adaptive(lin, Rstart, run);   // R->time_interval,
// ->time_point Inputs:   lin - the linearization (A, the centered input set, the constant input)
//           Rstart - the start set relative to x_star; run - the step size and the options
// Outputs:  R - the sets relative to x_star, or none
// See also: priv_lin_reach_adaptive, LinearSys::error_solution_adaptive
//
// CORA: contDynamics/@linearSys/initReach_adaptive.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       10-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys/private/priv_adaptive.h"
#include "cora/global/within_tol.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The Taylor series ------------------------------------------------------------------------------

/// The series of A: the powers, the remainder E of the series of e^{A dt}, and the interval
/// matrix F of the curvature of the state (CORA's aux_expmtie_adaptive).
struct Series {
    std::vector<Tensor> power;     ///< A^1, ..., A^(terms + 1) at index 0, ...
    std::vector<double> dt_by_fac; ///< dt^k / k! at index k - 1, up to terms + 1
    Interval E, F;
    int terms = 0;
};

double aux_dt_by_fac(double dt, int k) {
    double fac = 1;
    for (int j = 2; j <= k; ++j) fac *= j;
    return std::pow(dt, k) / fac;
}

/// The Taylor series of A dt: none if it has not converged after 50 terms.
std::optional<Series> aux_expmtie(const Tensor &A, double dt, std::optional<int> given,
                                  double zeta_t_lin) {
    const Tensor A_abs = A.abs();
    Series s{{A},
             {},
             Interval(A.zeros_like(), A.zeros_like()),
             Interval(A.zeros_like(), A.zeros_like()),
             0};
    std::vector<Tensor> power_abs{A_abs};
    Tensor M = A.eye_like(), sum_pos = A.zeros_like(), sum_neg = A.zeros_like();
    std::vector<double> norm_fro{0.0, 0.0}; // CORA: normAfro(1) is never set, it pads to 0
    for (int eta = 1;; ++eta) {
        // the powers of A and |A|, and dt^eta / eta!
        s.power.push_back(s.power.back().matmul(A));
        power_abs.push_back(power_abs.back().matmul(A_abs));
        s.dt_by_fac.push_back(aux_dt_by_fac(dt, eta));
        M = M + power_abs[std::size_t(eta - 1)] * s.dt_by_fac.back();
        if (eta == 1) continue; // F starts at eta = 2
        // the curvature: A^eta split into its positive and negative part
        const double e = eta;
        const double factor = (std::pow(e, -e / (e - 1)) - std::pow(e, -1 / (e - 1))) *
                              s.dt_by_fac[std::size_t(eta - 1)];
        const Tensor &P = s.power[std::size_t(eta - 1)];
        sum_pos = sum_pos + P.minimum(0.0) * factor;
        sum_neg = sum_neg + P.maximum(0.0) * factor;
        norm_fro.push_back((sum_pos - sum_neg).matrix_norm().item());
        // the series ends if A^eta is 0, the curvature stops growing, or at the given number
        const bool nilpotent = P.abs().sum().item() == 0.0;
        const bool stop =
            given ? eta == *given
                  : (nilpotent ||
                     1 - norm_fro[std::size_t(eta - 1)] / norm_fro[std::size_t(eta)] < zeta_t_lin);
        if (stop) {
            const Tensor W = ((A_abs * dt).expm() - M).abs();
            s.E = Interval(W * -1.0, W);
            s.F = Interval(sum_neg, sum_pos).plus(s.E);
            s.dt_by_fac.push_back(aux_dt_by_fac(dt, eta + 1));
            s.terms = eta;
            return s;
        }
        if (eta + 1 > 50) return std::nullopt;
    }
}

// The input ---------------------------------------------------------------------------------------

/// The correction of the input for a constant input: sum of the curvature matrices of A^i
/// weighted by the factor of eta = i + 1 (CORA's aux_inputTie), plus E dt.
Interval aux_input_tie(const Series &s, double dt) {
    Tensor sum_pos = s.E.inf().zeros_like(), sum_neg = sum_pos;
    for (int i = 2; i <= s.terms + 1; ++i) {
        const double e = i;
        const double factor = (std::pow(e, -e / (e - 1)) - std::pow(e, -1 / (e - 1))) *
                              s.dt_by_fac[std::size_t(i - 1)];
        const Tensor &P = s.power[std::size_t(i - 2)];
        sum_pos = sum_pos + P.minimum(0.0) * factor;
        sum_neg = sum_neg + P.maximum(0.0) * factor;
    }
    return Interval(sum_neg, sum_pos).plus(Interval(s.E.inf() * dt, s.E.sup() * dt));
}

/// Whether a set has no extent (CORA's representsa(V, 'origin', eps)).
bool aux_is_origin(const Zonotope &V) {
    const double eps = machine_eps();
    for (const double v : V.center().data())
        if (!within_tol(v, 0.0, eps)) return false;
    for (const double v : V.generators().data())
        if (!within_tol(v, 0.0, eps)) return false;
    return true;
}

} // namespace


// ===========================================  MAIN  =========================================== //

Zonotope priv_reduce_cached(const Zonotope &Z, double diag_percent, IndexCache &cache, int step) {
    if (cache.has(step)) return Z.reduce_idx(cache.at(step));
    AdaptiveReduction r = Z.reduce_adaptive(diag_percent);
    cache.set(step, std::move(r.idx));
    return std::move(r.Z);
}

std::optional<AdaptiveLinear> priv_init_reach_adaptive(const Linearized &lin,
                                                       const Zonotope &Rstart, AdaptiveRun &run) {
    const Tensor &A = lin.sys.A();
    const int64_t n = A.shape()[0];
    const double dt = run.time_step;
    const alg::LinearizationAdaptive &alg = run.alg;

    // the series, with the number of terms of the pass before in this step if there is one
    const std::optional<int> given =
        run.tt_lin.has(run.i) ? std::optional<int>(run.tt_lin.at(run.i)) : std::nullopt;
    const std::optional<Series> series = aux_expmtie(A, dt, given, alg.zeta_t_lin);
    if (!series) return std::nullopt;
    const Series &s = *series;
    run.tt_lin.set(run.i, s.terms);

    // the input: the time-varying part V = U, the constant part u_const
    const Zonotope &V = lin.U;
    const bool is_RV = !aux_is_origin(V);
    Tensor Asum = A.eye_like() * dt;
    std::optional<Zonotope> RV;
    if (is_RV) {
        Zonotope Vsum = V * dt;
        for (int k = 1; k <= s.terms; ++k) {
            const Tensor scaled = s.power[std::size_t(k - 1)] * s.dt_by_fac[std::size_t(k)];
            Vsum = Vsum.plus(V.linear_map(scaled));
            Asum = Asum + scaled;
        }
        const Interval E_dt(s.E.inf() * dt, s.E.sup() * dt);
        RV = Vsum.plus(V.linear_map(E_dt));
    } else {
        for (int k = 1; k <= s.terms; ++k)
            Asum = Asum + s.power[std::size_t(k - 1)] * s.dt_by_fac[std::size_t(k)];
    }
    // the solution of the constant input, and its correction
    const Zonotope v_trans(lin.u_const, lin.u_const.zeros_like({n, 1}));
    const Interval E_dt(s.E.inf() * dt, s.E.sup() * dt);
    const Interval e_at_int(Asum + E_dt.inf(), Asum + E_dt.sup());
    const Zonotope Rtrans = v_trans.linear_map(e_at_int);
    const Zonotope input_corr = v_trans.linear_map(aux_input_tie(s, dt));

    // the homogeneous solution, its enclosure over the step, and the reductions
    const Tensor e_at = (A * dt).expm();
    const Zonotope Rhom_tp = Rstart.linear_map(e_at).plus(Rtrans);
    const Zonotope Rhom = Rstart.enclose(Rhom_tp).plus(Rstart.linear_map(s.F)).plus(input_corr);
    Zonotope ti = priv_reduce_cached(Rhom, alg.zeta_z, run.idx_hom_ti, run.i);
    Zonotope tp = priv_reduce_cached(Rhom_tp, alg.zeta_z, run.idx_hom_tp, run.i);
    if (RV) {
        const Zonotope red = priv_reduce_cached(*RV, alg.zeta_z, run.idx_par, run.i);
        ti = ti.plus(red);
        tp = tp.plus(red);
    }
    return AdaptiveLinear{std::move(ti), std::move(tp)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
