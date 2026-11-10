// priv_lin_reach_adaptive - one step of the adaptive linearization algorithm
//
// As CORA's linReach_adaptive for alg "lin" (Wetzlinger et al. [1], Alg. 2). The step size is
// tuned from the gain of the abstraction error: a first pass at the horizon h, whose size follows
// from the gain of the step before, gives the error that a step of size h would add; a second
// pass at the size that minimizes the estimated width of the next sets (priv_optimal_delta_t)
// gives the gain of the step. Within a pass the admissible linearization error is enlarged by
// 1.1 until it holds; if it does not converge or the series of e^{A dt} does not, the step size
// is halved. From the second step on the tensor order may switch between 2 and 3, by the gain it
// would bring. All of it is CORA's, including its conventions for the arrays of the options.
//
// Syntax:   AdaptiveStep s = priv_lin_reach_adaptive(sys, Rstart, run);
// Inputs:   Rstart - the set at the start of the step
//           run - the options of the analysis, updated by the step
// Outputs:  s - the sets over the step and at its end (not yet reduced)
// References:
//    [1] M. Wetzlinger, A. Kulmburg, A. Le Penven and M. Althoff. "Adaptive Reachability
//        Algorithms for Nonlinear Systems Using Abstraction Error Analysis", NAHS, 2022.
// See also: priv_reach_adaptive, priv_init_reach_adaptive, priv_abstrerr_adaptive
//
// CORA: contDynamics/@nonlinearSys/linReach_adaptive.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       10-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_steps.h"
#include "cont_dynamics/nonlinear_sys/private/priv_adaptive.h"
#include "cora/global/within_tol.h"

#include <algorithm>
#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

/// The number of terms of the series of the error that the step uses, if it has fixed it.
std::optional<int> aux_error_terms(const AdaptiveRun &run) {
    return run.tt_err.has(run.i) ? std::optional<int>(run.tt_err.at(run.i)) : std::nullopt;
}

// The orders of the powers ------------------------------------------------------------------------

/// The Hessians as (n + m)^2 matrices, row-major: of the abstraction error (|H|, when constant)
/// or at the expansion point.
std::vector<std::vector<double>> aux_hessians(const NonlinearSys &sys, const AdaptiveRun &run,
                                              const Tensor &x_star,
                                              const std::optional<Tensor> &u_star) {
    const int64_t n = sys.dim(), d = n + sys.input_dim();
    std::vector<std::vector<double>> out;
    // the Hessian is |H| of the abstraction error if constant, else that at the point
    if (run.hessian_const) {
        for (int64_t i = 0; i < n; ++i) {
            std::vector<double> H(std::size_t(d * d), 0.0);
            for (std::size_t jk = 0; jk < std::size_t(d * d); ++jk)
                if (const auto &r = run.hessian_value[std::size_t(i * d * d) + jk])
                    H[jk] = std::max(std::abs(r->lo.item()), std::abs(r->hi.item()));
            out.push_back(std::move(H));
        }
    } else {
        for (const Tensor &H : sys.hessian(x_star, u_star)) out.push_back(H.data());
    }
    return out;
}

/// Which orders of the abstraction error the dimensions have in the powers of A (CORA's
/// aux_getPowers): the state dimensions with zero width and the Hessian say how soon the error
/// of a dimension reaches the others.
void aux_get_powers(const NonlinearSys &sys, AdaptiveRun &run, const Linearized &lin,
                    const std::vector<char> &zero_width, const Zonotope &V_dyn) {
    if (run.i == 1 && !run.orders.empty()) {
        run.min_order = int(*std::min_element(run.orders.begin(), run.orders.end()));
        return;
    }
    const int64_t n = sys.dim(), d = n + sys.input_dim();
    const Tensor &A = lin.sys.A();
    const std::optional<Tensor> u_star =
        sys.input_dim() > 0 ? std::optional<Tensor>(lin.u_star) : std::nullopt;
    // sigma: how many powers of A it takes for the error to matter in each dimension
    std::vector<double> sigma(std::size_t(n), 0.0);
    const bool all_zero =
        std::all_of(zero_width.begin(), zero_width.end(), [](char z) { return z; });
    const bool none_zero =
        std::none_of(zero_width.begin(), zero_width.end(), [](char z) { return z; });
    if (all_zero) {
        std::fill(sigma.begin(), sigma.end(), 2.0);
    } else if (!none_zero) {
        // the Hessian says which dimensions the error of a dimension enters
        const auto Hess = aux_hessians(sys, run, lin.x_star, u_star);
        for (int64_t i = 0; i < n; ++i) {
            const std::vector<double> &H = Hess[std::size_t(i)];
            auto at = [&](int64_t j, int64_t k) { return H[std::size_t(j * d + k)]; };
            // a dimension with a width and a diagonal entry of the Hessian: the error enters
            bool found = false;
            for (int64_t j = 0; j < n && !found; ++j)
                if (!zero_width[std::size_t(j)] && at(j, j) != 0) {
                    sigma[std::size_t(i)] = 0;
                    found = true;
                }
            if (found) continue;
            // otherwise the pairs of dimensions that couple: the fewer have a width, the sooner
            sigma[std::size_t(i)] = kInf;
            for (int64_t j = 0; j < n; ++j)
                for (int64_t jj = j + 1; jj < n; ++jj)
                    if (at(j, jj) + at(jj, j) != 0) {
                        sigma[std::size_t(i)] = std::min(sigma[std::size_t(i)],
                                                         double(int(zero_width[std::size_t(j)]) +
                                                                int(zero_width[std::size_t(jj)])));
                        found = true;
                    }
            // no pair couples: the error needs two powers
            if (!found) sigma[std::size_t(i)] = 2;
        }
    }
    // the dimensions whose error is nonzero, and the least power at which each one is reached
    const std::vector<double> rows = V_dyn.generators().abs().sum(-1, true).data();
    // CORA stops with an index error when the error is 0 everywhere, as for a linear system
    if (std::all_of(rows.begin(), rows.end(), [](double r) { return r == 0; }))
        CORA_FAIL(Error::not_supported,
                  "reach: the linearization error is 0 in every dimension (the system is linear); "
                  "the adaptive algorithm tunes the step to that error, use LinearSys::reach");
    // the least power of A at which the error of each dimension arrives
    std::vector<double> qi(std::size_t(n), kInf), a_power = A.eye_like().data();
    for (int p = 0;;) {
        for (std::size_t i = 0; i < std::size_t(n); ++i) {
            double least = kInf;
            for (std::size_t j = 0; j < std::size_t(n); ++j)
                if (rows[j] != 0 && a_power[i * std::size_t(n) + j] != 0)
                    least = std::min(least, sigma[j]);
            qi[i] = std::min(qi[i], least + p);
        }
        // done when every dimension is reached by the powers up to p
        if (std::all_of(qi.begin(), qi.end(), [&](double q) { return q < p + 1; })) break;
        ++p;
        a_power = Tensor::like(A, a_power, A.shape()).matmul(A).data();
        if (p == 100) {
            warn("CORA:contDynamics", "Check computation for order of abstraction error");
            if (std::all_of(qi.begin(), qi.end(),
                            [&](double q) { return std::isinf(q) || q < p + 1; }))
                break;
        }
    }
    const double least = *std::min_element(qi.begin(), qi.end());
    run.orders = qi;
    run.min_order = int(least);
}

// The tensor order --------------------------------------------------------------------------------

/// The tensor order of the first step: 3 if its error is below zeta_k of that of order 2 in some
/// dimension, else 2 (CORA's aux_initStepTensorOrder).
void aux_init_step_tensor_order(const NonlinearSys &sys, const Zonotope &Rstart, AdaptiveRun &run) {
    if (Rstart.generators().numel() == 0) {
        run.tensor_order = 2;
        return;
    }
    // the errors of order 2 and 3 over the start set, expanded at its center and the center of U0
    const Tensor &x_star = Rstart.center();
    const std::optional<Tensor> u_star =
        run.U0 ? std::optional<Tensor>(run.U0->center()) : std::nullopt;
    const Zonotope Rdelta = Rstart - x_star;
    run.tensor_order = 2;
    const std::vector<double> L2 =
        priv_abstrerr_adaptive(sys, Rdelta, x_star, u_star, run).bound.data();
    run.tensor_order = 3;
    const std::vector<double> L3 =
        priv_abstrerr_adaptive(sys, Rdelta, x_star, u_star, run).bound.data();
    // the dimensions where the errors are nonzero
    std::vector<double> nonzero2, nonzero3;
    for (const double v : L2)
        if (v != 0) nonzero2.push_back(v);
    for (const double v : L3)
        if (v != 0) nonzero3.push_back(v);
    if (nonzero2.size() != nonzero3.size())
        CORA_FAIL(Error::dimension_mismatch,
                  "reach: the abstraction errors of order 2 and 3 are nonzero in different "
                  "dimensions");
    bool all = true;
    for (std::size_t j = 0; j < nonzero2.size(); ++j)
        all = all && nonzero3[j] / nonzero2[j] > run.alg.zeta_k;
    run.tensor_order = all ? 2 : 3;
}

/// Whether the other tensor order would pay off since the last switch, and the switch if so
/// (CORA's aux_nextStepTensorOrder). The errors it computes are tested, not kept.
void aux_next_step_tensor_order(const NonlinearSys &sys, AdaptiveRun &run, double abstrerr,
                                const Linearized &lin, const Zonotope &Rmax) {
    const std::optional<Tensor> u_star =
        sys.input_dim() > 0 ? std::optional<Tensor>(lin.u_star) : std::nullopt;
    const double mu = run.alg.decr_factor;
    const double varphi_lim = std::pow(mu, run.min_order + 1);
    const double total =
        priv_varphitotal(run.varphi.at(run.i), varphi_lim, run.kappa_delta_t, run.time_step, mu);
    // is the error since the last switch moving?
    const double estimate = total * abstrerr;
    if (!(std::abs(estimate / run.kappa_abstrerr - 1) > 1 - run.alg.zeta_k)) return;

    // the error of the other order at this step size
    const int old_order = run.tensor_order;
    run.tensor_order = old_order == 2 ? 3 : 2;
    AdaptiveRun scratch = run;
    const AbstractionError other = priv_abstrerr_adaptive(sys, Rmax, lin.x_star, u_star, scratch);
    int used = 0;
    const Zonotope Rerror =
        priv_error_solution_adaptive(lin.sys.A(), other.set, std::nullopt, run.time_step,
                                     run.alg.zeta_t_abs, aux_error_terms(run), used);
    const double abstrerr_other = priv_vector_norm(priv_row_sums(Rerror));
    // 2 -> 3 keeps 3 if the new error is below zeta_k of the old; 3 -> 2 keeps 2 if it is above
    const double gain = old_order == 2 ? abstrerr_other / abstrerr : abstrerr / abstrerr_other;
    const bool switch_order = old_order == 2 ? !(gain > run.alg.zeta_k) : gain > run.alg.zeta_k;
    if (switch_order) {
        run.kappa_delta_t = run.time_step;
        run.kappa_abstrerr = abstrerr_other;
        run.error_adm_delta_t_opt = other.bound;
        run.error_adm_horizon = other.bound.zeros_like();
    } else {
        run.tensor_order = old_order;
    }
}

// The passes --------------------------------------------------------------------------------------

/// What the pass at the horizon leaves for the pass at the step size (CORA's variables *_h).
struct HorizonPass {
    std::optional<Zonotope> time_interval, time_point, error;
    std::vector<double> abstrerr;
    Tensor x_star;
    double zeta_p = 0;
};

/// The linearization error of a pass and the set it was computed over.
struct Admissible {
    AbstractionError error;
    Zonotope R_max;
};

/// The largest ratio of the abstraction error of this pass to that of the pass at the horizon, over
/// the dimensions of the minimal order (MATLAB's max skips NaN); NaN if there is none.
double aux_gain_ratio(const std::vector<double> &abstrerr, const std::vector<double> &horizon,
                      const AdaptiveRun &run) {
    const std::vector<double> now = priv_at_min_order(abstrerr, run);
    const std::vector<double> then = priv_at_min_order(horizon, run);
    double gain = std::numeric_limits<double>::quiet_NaN();
    for (std::size_t j = 0; j < now.size(); ++j) {
        const double ratio = now[j] / then[j];
        if (!std::isnan(ratio) && (std::isnan(gain) || ratio > gain)) gain = ratio;
    }
    return gain;
}

// The horizon -------------------------------------------------------------------------------------

/// The size of the horizon of a step: 1 % of the time span at first, the rest of the time at the
/// end, else from the gain of the step before; the flags say which.
double aux_horizon(AdaptiveRun &run, bool &very_first, bool &last_step) {
    if (run.i == 1) {
        very_first = true;
        run.time_step = run.t_final * 0.01;
        run.tt_err.clear();
        run.idx_hom_ti.clear();
        run.idx_hom_tp.clear();
        run.idx_par.clear();
        run.idx_red.clear();
        run.idx_err.clear();
        return run.time_step;
    }
    if (run.time_step > run.t_final - run.t) {
        last_step = true;
        run.time_step = run.t_final - run.t;
        return run.time_step;
    }
    // the horizon of the step before, grown by the gain over the limit of its order
    double horizon =
        run.finite_horizon.at(run.i - 1) * (1 + run.varphi.at(run.i - 1) - priv_zeta_phi(run));
    if (within_tol(horizon, run.step_size.at(run.i - 1), numerics::kHorizonMatch))
        horizon = run.finite_horizon.at(run.i - 1) * 1.1;
    run.time_step = horizon;
    if (!(run.time_step > 0))
        CORA_FAIL(Error::report_to_dev, "reach: tuning error (the step size is not positive)");
    return horizon;
}

// The error of a pass -----------------------------------------------------------------------------

/// The linearization error of a pass, with the admissible error enlarged by 1.1 until it holds;
/// none if it does not converge (the error rises again, or leaves the domain of the dynamics).
std::optional<Admissible> aux_admissible_error(const NonlinearSys &sys, AdaptiveRun &run,
                                               const Linearized &lin, const AdaptiveLinear &Rlin,
                                               Tensor &error_adm) {
    const std::optional<Tensor> u_star =
        sys.input_dim() > 0 ? std::optional<Tensor>(lin.u_star) : std::nullopt;
    std::vector<double> perf{0.0};
    for (int counter = 1;; ++counter) {
        // the solution of the admissible error moves the set of the pass
        Zonotope R_all_error(lin.x_star.zeros_like(), lin.x_star.zeros_like({sys.dim(), 0}));
        const std::vector<double> adm = error_adm.data();
        if (std::any_of(adm.begin(), adm.end(), [](double v) { return v != 0; })) {
            // the columns of the nonzero entries of the admissible error
            std::vector<int64_t> cols;
            for (std::size_t j = 0; j < adm.size(); ++j)
                if (adm[j] != 0) cols.push_back(int64_t(j));
            const Zonotope V_error(error_adm.zeros_like(), error_adm.diag().take(cols, -1));
            int used = 0;
            R_all_error =
                priv_error_solution_adaptive(lin.sys.A(), V_error, std::nullopt, run.time_step,
                                             run.alg.zeta_t_abs, aux_error_terms(run), used);
            run.tt_err.set(run.i, used);
        }
        // the set of the pass with the error, and the error over it
        Zonotope R_max = Rlin.time_interval + R_all_error;
        std::optional<AbstractionError> err;
        try {
            err = priv_abstrerr_adaptive(sys, R_max, lin.x_star, u_star, run);
        } catch (const Error &e) {
            if (e.id() == Error::out_of_domain) return std::nullopt;
            CORA_FAIL(e.id(), e.what());
        }
        // the error holds, or it rises again, or the admissible error is enlarged
        const double current = priv_ratio_max(err->bound, error_adm);
        const std::vector<double> bound = err->bound.data();
        if (current <= 1 ||
            std::none_of(bound.begin(), bound.end(), [](double v) { return v != 0; }))
            return Admissible{std::move(*err), std::move(R_max)};
        if (counter > 1) {
            perf.resize(std::size_t(counter) + 1);
            perf[std::size_t(counter)] = current;
            if (counter > 2 && perf[std::size_t(counter)] > perf[std::size_t(counter) - 1])
                return std::nullopt;
        }
        error_adm = err->bound * 1.1;
    }
}

/// Remembers the pass at the horizon.
void aux_save_horizon(HorizonPass &h, const std::vector<double> &abstrerr, const Zonotope &error,
                      const AdaptiveLinear &Rlin, const Linearized &lin) {
    h.abstrerr = abstrerr;
    h.error = error;
    h.time_interval = Rlin.time_interval;
    h.time_point = Rlin.time_point;
    h.x_star = lin.x_star;
}

} // namespace


// ===========================================  MAIN  =========================================== //

AdaptiveStep priv_lin_reach_adaptive(const NonlinearSys &sys, const Zonotope &Rstart,
                                     AdaptiveRun &run) {
    const double mu = run.alg.decr_factor;
    Tensor error_adm = run.error_adm_horizon;
    bool last_step = false, very_first = false, equal_horizon = false;
    double finite_horizon = aux_horizon(run, very_first, last_step);

    // the dimensions in which the start set has no extent (CORA's withinTol, tolerance 1e-8)
    std::vector<char> zero_width;
    for (const double r : priv_row_sums(Rstart))
        zero_width.push_back(within_tol(r, 0.0, numerics::kZeroWidth));
    run.pass = 0;
    if (very_first) aux_init_step_tensor_order(sys, Rstart, run);
    run.pass = 1;

    // what the pass at the horizon leaves for the pass at the step size
    HorizonPass h;
    std::optional<Linearized> lin;
    std::optional<AdaptiveLinear> Rlin;
    std::optional<Zonotope> Rerror;
    double zeta_p = 0;
    for (;;) {
        // the passes: at the horizon, then at the step size it gives
        if (!(run.time_step > numerics::kMinTimeStep))
            CORA_FAIL(Error::not_converged,
                      "reach: the step size of the adaptive algorithm converged to 0 at t = " +
                          std::to_string(run.t));
        lin = sys.linearize(Rstart, run.U, run.time_step);
        if (run.pass == 1) zeta_p = std::exp((lin->sys.A() * run.time_step).trace().item());
        Rlin = priv_init_reach_adaptive(*lin, Rstart - lin->x_star, run);
        if (!Rlin) { // the series of e^{A dt} does not converge: the step size is too large
            run.time_step *= 0.5;
            finite_horizon = run.time_step;
            continue;
        }
        std::optional<Admissible> adm = aux_admissible_error(sys, run, *lin, *Rlin, error_adm);
        if (!adm) {
            run.time_step *= 0.5;
            finite_horizon = run.time_step;
            error_adm = run.error_adm_horizon;
            continue;
        }
        // the powers of A that matter, when the start set or the first step changes them
        const AbstractionError &err = adm->error;

        // the powers of A that matter, when the start set or the first step changes them
        if (very_first || !run.zero_width || *run.zero_width != zero_width) {
            run.zero_width = zero_width;
            aux_get_powers(sys, run, *lin, zero_width, err.set);
        }
        int used = 0;
        Rerror = priv_error_solution_adaptive(lin->sys.A(), err.set, std::nullopt, run.time_step,
                                              run.alg.zeta_t_abs, aux_error_terms(run), used);
        const std::vector<double> abstrerr = priv_row_sums(*Rerror);

        if (run.pass == 1 && !last_step) {
            // the tuning of the step size from the pass at the horizon
            if (run.i == 1) {
                const bool first = very_first;
                very_first = false;
                const double varphi = first ? 0.0 : aux_gain_ratio(abstrerr, h.abstrerr, run);
                if (std::isnan(varphi))
                    CORA_FAIL(Error::special_error,
                              "reach: the abstraction error of the first step is zero");
                if (first || varphi < priv_zeta_phi(run)) { // a smaller horizon first
                    if (!first) finite_horizon = run.time_step;
                    aux_save_horizon(h, abstrerr, *Rerror, *Rlin, *lin);
                    h.zeta_p = zeta_p;
                    run.time_step = mu * run.time_step;
                    error_adm = err.bound * mu;
                    continue;
                }
                // the gain of the horizon, and the step size that minimizes the width after it
                run.varphi.set(run.i, varphi);
                run.finite_horizon.set(run.i, finite_horizon);
                run.time_step =
                    priv_optimal_delta_t(Rstart, *h.error, finite_horizon, varphi, h.zeta_p, run);
                run.error_adm_horizon = err.bound;
                error_adm = err.bound.zeros_like();
            } else {
                aux_save_horizon(h, abstrerr, *Rerror, *Rlin, *lin);
                run.time_step = priv_optimal_delta_t(Rstart, *h.error, finite_horizon,
                                                     run.varphi.at(run.i - 1), zeta_p, run);
                run.time_step = std::min(run.time_step, run.t_final - run.t);
                run.error_adm_horizon = err.bound;
                error_adm = run.error_adm_delta_t_opt;
            }
        } else if (run.pass == 2 || last_step) {
            // the gain of the step, and whether the tensor order should change
            if (equal_horizon)
                run.varphi.set(run.i, aux_gain_ratio(abstrerr, h.abstrerr, run));
            else if (!last_step)
                run.varphi.set(
                    run.i, priv_varphi_est(finite_horizon, run.time_step, *h.error, *Rerror, run));
            run.finite_horizon.set(run.i, finite_horizon);
            run.error_adm_delta_t_opt = err.bound;
            if (run.i == 1) {
                run.kappa_delta_t = run.time_step;
                run.kappa_abstrerr = priv_vector_norm(abstrerr);
            } else if (!last_step) {
                aux_next_step_tensor_order(sys, run, priv_vector_norm(abstrerr), *lin, adm->R_max);
            }
            break;
        }

        // the second pass: at the step size the first one computed
        ++run.pass;
        if (run.time_step == finite_horizon) {
            equal_horizon = true;
            run.time_step = run.time_step * mu;
        }
    }

    // the sets of the step: of the horizon pass if its size was kept, else of the last pass
    Zonotope Rti = Rlin->time_interval + lin->x_star;
    Zonotope Rtp = Rlin->time_point + lin->x_star;
    if (equal_horizon) {
        run.time_step = finite_horizon;
        Rti = *h.time_interval + h.x_star;
        Rtp = *h.time_point + h.x_star;
        Rerror = h.error;
    }
    run.step_size.set(run.i, run.time_step);
    return {Rti + *Rerror, Rtp + *Rerror};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
