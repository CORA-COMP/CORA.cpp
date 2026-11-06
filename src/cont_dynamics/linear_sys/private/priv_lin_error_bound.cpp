// priv_lin_error_bound - the error bookkeeping of the adaptive reachability algorithm
//
// As CORA's linErrorBound.
// The methods of LinErrorBound: bounds per step, accumulation of the committed errors, the
// consistency check at the end, the allocation of emax to the reduction error, and the prediction
// of the next time step size by approximation functions or bisection.
//
// Syntax:   LinErrorBound errs(emax, t_final);   errs.next_bounds(time_step, t, k, kIter);
// Inputs:   emax - error margin
//           t_final - time horizon
// Outputs:  R - the bounds and checks of priv_linErrorBound.h
// See also: priv_reach_adaptive
//
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   03-November-2026
// Reviewed:      ---

#include "cont_dynamics/linear_sys/private/priv_lin_error_bound.h"
#include "cora/global/error.h"
#include "cora/global/tensor/tolerance.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

/// Sets v[k][it], growing the lists as MATLAB does when an element is assigned.
template <class T>
void aux_put(std::vector<std::vector<T>> &v, int k, int it, T val) {
    if (static_cast<int>(v.size()) <= k) v.resize(k + 1);
    if (static_cast<int>(v[k].size()) <= it) v[k].resize(it + 1, T());
    v[k][it] = val;
}

/// The smallest and largest of the values that are not NaN (MATLAB's min and max).
double aux_min(std::initializer_list<double> values) {
    double m = std::numeric_limits<double>::infinity();
    bool any = false;
    for (double v : values)
        if (!std::isnan(v)) m = any ? std::min(m, v) : v, any = true;
    return any ? m : kNaN;
}

double aux_max(std::initializer_list<double> values) {
    double m = -std::numeric_limits<double>::infinity();
    bool any = false;
    for (double v : values)
        if (!std::isnan(v)) m = any ? std::max(m, v) : v, any = true;
    return any ? m : kNaN;
}

/// Whether any entry of the list is below (or above) the limit.
bool aux_any_below(const std::vector<double> &v, double limit) {
    for (double x : v)
        if (x < limit) return true;
    return false;
}

bool aux_any_above(const std::vector<double> &v, double limit) {
    for (double x : v)
        if (x > limit) return true;
    return false;
}

/// The condition number (2-norm) of the matrix [a b; c d], from its singular values.
double aux_cond2x2(double a, double b, double c, double d) {
    const double frob2 = a * a + b * b + c * c + d * d;
    const double det = std::abs(a * d - b * c);
    const double plus = std::sqrt(std::max(frob2 + 2 * det, 0.0));
    const double minus = std::sqrt(std::max(frob2 - 2 * det, 0.0));
    return (plus + minus) / (plus - minus);
}

/// The solution of [a b; c d] x = r by LU with partial pivoting, as MATLAB's backslash.
void aux_solve2x2(double a, double b, double c, double d, double r1, double r2, double &x1,
                  double &x2) {
    if (std::abs(c) > std::abs(a)) {
        std::swap(a, c);
        std::swap(b, d);
        std::swap(r1, r2);
    }
    const double l = c / a;
    const double u22 = d - l * b;
    const double y2 = r2 - l * r1;
    x2 = y2 / u22;
    x1 = (r1 - b * x2) / a;
}

/// The 2-norm of the column sums |V| of a matrix (n, m): MATLAB's vecnorm(sum(abs(V), 2)).
double aux_row_sum_norm(const Tensor &V) {
    double s = 0;
    for (double v : V.abs().sum(-1, true).data()) s += v * v;
    return std::sqrt(s);
}

} // namespace


// ===========================================  MAIN  =========================================== //

// Bounds and bookkeeping --------------------------------------------------------------------------

LinErrorBound::LinErrorBound(double emax_, double t_final_) : emax(emax_), t_final(t_final_) {
    if (!std::isfinite(emax_) || emax_ <= 0)
        CORA_FAIL(Error::wrong_value, "linErrorBound: the error margin must be a positive number");
    if (!std::isfinite(t_final_) || t_final_ < 0)
        CORA_FAIL(Error::wrong_value, "linErrorBound: the time horizon must be non-negative");
}

void LinErrorBound::next_bounds(double time_step, double t, int k, int kIter) {
    aux_put(time_steps, k, kIter, time_step);
    // The margin of the step: emax without the final reduction error and the error accumulated.
    if (static_cast<int>(bound_remacc.size()) <= k) bound_remacc.resize(k + 1);
    bound_remacc[k] = k == 0 ? emax - bound_red_max : emax - bound_red_max - cum_acc[k - 1];
    // The accumulating error may use the share of the margin that the step has of the time left.
    aux_put(bound_acc, k, kIter, bound_remacc[k] * time_step / (t_final - t));
    const double bound_red_tk = bound_red_max * (t + time_step) / t_final;
    if (k == 0) {
        aux_put(bound_red, k, kIter, bound_red_tk);
        aux_put(bound_rem, k, kIter, emax - bound_red_tk);
    } else {
        aux_put(bound_red, k, kIter, bound_red_tk - cum_red[k - 1]);
        aux_put(bound_rem, k, kIter, emax - bound_red_tk - cum_acc[k - 1]);
    }
}

void LinErrorBound::accumulate_errors(int k, int kIter) {
    if (k == 0) {
        cum_acc = {step_acc[k][kIter]};
        cum_red = {step_red[k]};
    } else {
        cum_acc.push_back(cum_acc.back() + step_acc[k][kIter]);
        cum_red.push_back(cum_red.back() + step_red[k]);
    }
}

void LinErrorBound::remove_redundant_values(int k, int kIter) {
    for (PerStep *v : {&seq_nonacc, &idv_pu_tkplus1, &idv_f, &idv_g, &idv_lin_comb, &idv_pu_tauk,
                       &step_acc, &bound_rem, &bound_acc, &bound_red})
        (*v)[k] = {(*v)[k][kIter]};
    // The step size is the last per-step list.
    time_steps[k] = {time_steps[k][kIter]};
}

void LinErrorBound::full_errors(int k, double &RcontError, double &RcontTpError) const {
    // The non-accumulating error contains the accumulating error of the last step.
    RcontError =
        k == 0 ? seq_nonacc[k][0] + cum_red[k] : seq_nonacc[k][0] + cum_acc[k - 1] + cum_red[k];
    RcontTpError = cum_acc[k] + cum_red[k];
}

// Checks ------------------------------------------------------------------------------------------

bool LinErrorBound::check_errors(const std::vector<double> &RoutError,
                                 const std::vector<double> &RoutTpError, bool &internal_ok) const {
    bool res = true;
    internal_ok = true;
    const std::size_t K = time_steps.size();
    std::vector<double> t_vec, nonacc, acc, nonacc_bound, acc_bound, red_bound;
    for (std::size_t k = 0; k < K; ++k) {
        t_vec.push_back(time_steps[k][0]);
        nonacc.push_back(seq_nonacc[k][0]);
        acc.push_back(step_acc[k][0]);
        nonacc_bound.push_back(bound_rem[k][0]);
        acc_bound.push_back(bound_acc[k][0]);
        red_bound.push_back(bound_red[k][0]);
    }
    std::vector<char> full(K);
    for (std::size_t k = 0; k < K; ++k) full[k] = !std::isnan(nonacc[k]);
    std::vector<double> tp = RoutTpError;
    if (tp.empty()) tp.assign(K, 0.0);

    // The bounds and errors the committed values are checked against.
    std::vector<double> acc_total_bound(K), total_error(K), total_error_tp(K);
    for (std::size_t k = 0; k < K; ++k) {
        acc_total_bound[k] = cum_acc[k] + acc_bound[k] - acc[k];
        total_error[k] = nonacc[k] + cum_acc[k] - acc[k] + cum_red[k];
        total_error_tp[k] = cum_acc[k] + cum_red[k];
    }

    // All errors and bounds are non-negative; the full errors stay below emax.
    std::vector<double> full_err, full_tp, full_nonacc;
    for (std::size_t k = 0; k < K; ++k)
        if (full[k]) {
            full_err.push_back(RoutError[k]);
            full_tp.push_back(tp[k]);
            full_nonacc.push_back(nonacc[k]);
        }
    for (const std::vector<double> *v : std::initializer_list<const std::vector<double> *>{
             &full_err, &full_tp, &full_nonacc, &acc, &step_red, &nonacc_bound, &acc_bound,
             &red_bound, &acc_total_bound})
        if (aux_any_below(*v, 0.0)) res = false;
    if (aux_any_above(full_err, emax) || aux_any_above(full_tp, emax)) res = false;

    // The recomputed full errors match the ongoing computation.
    bool any_tp = false;
    for (double v : tp) any_tp = any_tp || v != 0;
    for (std::size_t k = 0; k < K; ++k) {
        if (!full[k]) continue;
        const double agree = numerics::kInternalConsistency;
        if (std::abs(total_error[k] - RoutError[k]) > agree) internal_ok = false;
        if (any_tp && std::abs(total_error_tp[k] - tp[k]) > agree) internal_ok = false;
    }

    // Non-accumulating errors, accumulating errors and their linearly increasing bound.
    if (aux_any_above(nonacc, emax)) res = false;
    for (std::size_t k = 0; k < K; ++k)
        if (nonacc[k] > nonacc_bound[k]) internal_ok = false;
    if (aux_any_above(cum_acc, emax) || aux_any_above(cum_acc, emax - bound_red_max)) res = false;
    for (std::size_t k = 0; k < K; ++k)
        if (acc[k] > cum_acc[k] || acc[k] > acc_bound[k]) internal_ok = false;
    const double final_value = emax - bound_red_max;
    double cum_t = 0;
    std::vector<double> cumsum_t(K);
    for (std::size_t k = 0; k < K; ++k) cumsum_t[k] = cum_t += t_vec[k];
    for (std::size_t k = 0; k < K; ++k) {
        const double lin_bound = final_value * cumsum_t[k] / cumsum_t[K - 1];
        if (cum_acc[k] > lin_bound || lin_bound - acc_total_bound[k] < -10 * machine_eps())
            internal_ok = false;
    }

    // Reduction errors.
    if (aux_any_above(cum_red, emax)) res = false;
    for (std::size_t k = 0; k < K; ++k)
        if (step_red[k] > cum_red[k] || step_red[k] > red_bound[k] || cum_red[k] > bound_red_max)
            internal_ok = false;
    return res;
}

// Reduction error ---------------------------------------------------------------------------------

void LinErrorBound::compute_error_bound_reduction(const Tensor &A, const Tensor &GU) {
    // No input generators: nothing to reduce.
    if (GU.abs().sum().item() == 0) {
        bound_red_max = 0;
        return;
    }
    const int steps_for_error_bound = 100;
    const double time_step = t_final / steps_for_error_bound;
    Tensor e_at = A.eye_like();
    const Tensor e_a_dt = (A * time_step).expm();

    // The errors of the auxiliary sets V_k = e^{A k dt} dt GU.
    const Tensor DtU = GU * time_step;
    std::vector<double> err_v(steps_for_error_bound);
    for (int i = 0; i < steps_for_error_bound; ++i) {
        const Tensor V = e_at.matmul(DtU);
        e_at = e_at.matmul(e_a_dt);
        err_v[i] = aux_row_sum_norm(V);
    }

    // Weights, ordering and the cumulative error of the ordered sets.
    double total = 0;
    for (double e : err_v) total += e;
    std::vector<int> tau(steps_for_error_bound);
    std::iota(tau.begin(), tau.end(), 0);
    std::stable_sort(tau.begin(), tau.end(),
                     [&](int a, int b) { return err_v[a] / total < err_v[b] / total; });
    std::vector<double> cumsum(steps_for_error_bound);
    double run = 0;
    for (int i = 0; i < steps_for_error_bound; ++i) cumsum[i] = run += err_v[tau[i]];

    // How much could be reduced if a share of emax were allocated to the reduction error?
    const int meshsize = 1000;
    const double d = 1.0 / (meshsize - 1);
    double best_final_order = steps_for_error_bound + 1;
    int min_idx = 1;
    for (int i = 1; i <= meshsize - 1; ++i) {
        const double pct = (i - 1) * d;
        int idx = 0;
        for (int j = steps_for_error_bound; j >= 1; --j)
            if (cumsum[j - 1] < emax * pct) {
                idx = j;
                break;
            }
        const double N = 1.0 / (1.0 - pct);
        const double final_order =
            idx == 0 ? N * (steps_for_error_bound + 1) : N * (steps_for_error_bound + 1 - idx);
        if (final_order < best_final_order) {
            best_final_order = final_order;
            // CORA takes the index into the mesh from the count of reducible sets.
            min_idx = idx;
        }
    }
    bound_red_max = emax * ((min_idx - 1) * d);
}

// Approximation functions -------------------------------------------------------------------------

void LinErrorBound::update_coefficients_approx_fun(int k, int kIter, bool fullcomp) {
    if (!use_approx_fun) return;
    const double dt = time_steps[k][kIter];
    if (kIter == 0) {
        // Initial guess: b = 0 for the quadratic functions.
        if (fullcomp) {
            coeff_lin_comb_b = idv_lin_comb[k][kIter] / dt;
            coeff_pu_tauk_b = idv_pu_tauk[k][kIter] / dt;
            coeff_f_a = idv_f[k][kIter] / (dt * dt);
            coeff_f_b = 0;
            coeff_g_a = idv_g[k][kIter] / (dt * dt);
            coeff_g_b = 0;
        }
        coeff_pu_tkplus1_a = idv_pu_tkplus1[k][kIter] / (dt * dt);
        coeff_pu_tkplus1_b = 0;
        return;
    }
    if (fullcomp) {
        coeff_lin_comb_b = idv_lin_comb[k][kIter] / dt;
        coeff_pu_tauk_b = idv_pu_tauk[k][kIter] / dt;
    }
    // a dt^2 + b dt through the two most recent values.
    const double dt_prev = time_steps[k][kIter - 1];
    const double m11 = dt_prev * dt_prev, m12 = dt_prev, m21 = dt * dt, m22 = dt;
    if (std::abs(1 / aux_cond2x2(m11, m12, m21, m22)) < machine_eps())
        CORA_FAIL(Error::special_error,
                  "LinearSys::verify: estimation of the time step size failed");
    aux_solve2x2(m11, m12, m21, m22, idv_pu_tkplus1[k][kIter - 1], idv_pu_tkplus1[k][kIter],
                 coeff_pu_tkplus1_a, coeff_pu_tkplus1_b);
    if (fullcomp) {
        aux_solve2x2(m11, m12, m21, m22, idv_f[k][kIter - 1], idv_f[k][kIter], coeff_f_a,
                     coeff_f_b);
        if (coeff_f_b < 0) {
            // A negative b cannot occur (the error is positive): use the last value only.
            coeff_f_b = 0;
            coeff_f_a = idv_f[k][kIter] / (dt * dt);
        }
        aux_solve2x2(m11, m12, m21, m22, idv_g[k][kIter - 1], idv_g[k][kIter], coeff_g_a,
                     coeff_g_b);
    }
}

// Bisection ---------------------------------------------------------------------------------------

void LinErrorBound::update_bisection(int k, int kIter, bool is_u, double time_step) {
    const bool acc_ok = bound_acc_ok[k][kIter], nonacc_ok = bound_nonacc_ok[k][kIter];
    const double acc_perc = step_acc[k][kIter] / bound_acc[k][kIter];
    const double nonacc_perc = seq_nonacc[k][kIter] / bound_rem[k][kIter];
    if (kIter == 0) {
        // The lower bound is 0; the upper bound is the first step size.
        bisect_lb_time_step_acc = bisect_lb_acc = bisect_lb_acc_perc = 0;
        bisect_lb_accok = true;
        bisect_lb_time_step_nonacc = bisect_lb_nonacc = bisect_lb_nonacc_perc = 0;
        bisect_lb_nonaccok = true;
        bisect_ub_time_step_acc = time_step;
        bisect_ub_acc = idv_pu_tkplus1[k][kIter];
        bisect_ub_accok = acc_ok;
        bisect_ub_acc_perc = acc_perc;
        bisect_ub_time_step_nonacc = time_step;
        bisect_ub_nonacc = seq_nonacc[k][kIter];
        bisect_ub_nonaccok = nonacc_ok;
        bisect_ub_nonacc_perc = nonacc_perc;
        return;
    }
    if (is_u) {
        if (!acc_ok || time_step > bisect_ub_time_step_acc) {
            // A step size that fails the check, or exceeds the bracket, is the new upper bound.
            if (time_step > bisect_ub_time_step_acc) {
                bisect_lb_time_step_acc = bisect_ub_time_step_acc;
                bisect_lb_acc = bisect_ub_acc;
                bisect_lb_accok = bisect_ub_accok;
                bisect_lb_acc_perc = bisect_ub_acc_perc;
            }
            bisect_ub_time_step_acc = time_step;
            bisect_ub_acc = idv_pu_tkplus1[k][kIter];
            bisect_ub_accok = acc_ok;
            bisect_ub_acc_perc = acc_perc;
        } else {
            bisect_lb_time_step_acc = time_step;
            bisect_lb_acc = idv_pu_tkplus1[k][kIter];
            bisect_lb_accok = acc_ok;
            bisect_lb_acc_perc = acc_perc;
        }
    }
    // The same for the non-accumulating error.
    if (!nonacc_ok || time_step > bisect_ub_time_step_nonacc) {
        if (time_step > bisect_ub_time_step_nonacc) {
            bisect_lb_time_step_nonacc = bisect_ub_time_step_nonacc;
            bisect_lb_nonacc = bisect_ub_nonacc;
            bisect_lb_nonaccok = bisect_ub_nonaccok;
            bisect_lb_nonacc_perc = bisect_ub_nonacc_perc;
        }
        bisect_ub_time_step_nonacc = time_step;
        bisect_ub_nonacc = seq_nonacc[k][kIter];
        bisect_ub_nonaccok = nonacc_ok;
        bisect_ub_nonacc_perc = nonacc_perc;
    } else {
        bisect_lb_time_step_nonacc = time_step;
        bisect_lb_nonacc = seq_nonacc[k][kIter];
        bisect_lb_nonaccok = nonacc_ok;
        bisect_lb_nonacc_perc = nonacc_perc;
    }
}

// Step size prediction ----------------------------------------------------------------------------

double LinErrorBound::approx_fun(double t, int k, int kIter, bool fullcomp, double time_step,
                                 double max_time_step) const {
    // The first guess of the first step may be far too large: shrink it before predicting.
    if (k == 0 && (seq_nonacc[k][kIter] / bound_rem[k][kIter] > 1e3 ||
                   step_acc[k][kIter] / bound_acc[k][kIter] > 1e3))
        return 0.01 * time_step;

    // The safety factor makes the bounds more likely to hold if the functions underestimate.
    const double safety_factor = 0.90;

    // 1. The accumulating error meets the linearly increasing bound up to the time horizon.
    const double pred_lin_bound =
        safety_factor *
        (1 / coeff_pu_tkplus1_a * (bound_remacc[k] / (t_final - t) - coeff_pu_tkplus1_b));

    // 2. All errors meet the remaining error of the step: a quadratic equation.
    double pred_erem = std::numeric_limits<double>::infinity();
    if (fullcomp) {
        const double a = coeff_pu_tkplus1_a + coeff_f_a + coeff_g_a;
        const double b =
            coeff_pu_tkplus1_b + coeff_f_b + coeff_g_b + coeff_lin_comb_b + coeff_pu_tauk_b;
        const double c = -bound_rem[k][kIter];
        const double root = std::sqrt(b * b - 4 * a * c);
        pred_erem = safety_factor * aux_max({1 / (2 * a) * (-b + root), 1 / (2 * a) * (-b - root)});
    }
    return aux_min({max_time_step, pred_lin_bound, pred_erem});
}

double LinErrorBound::bisection(double t, int k, bool fullcomp, double time_step,
                                double max_time_step, bool is_u) const {
    (void)time_step;
    const double eacc_total = k == 0 ? 0.0 : cum_acc[k - 1];
    const double inf = std::numeric_limits<double>::infinity();

    // 1. Accumulating error.
    double pred_eacc = inf, slope_acc = 0;
    if (is_u) {
        if (bisect_ub_accok) {
            // Extrapolate the linear trend of the error up to the linear bound.
            const double bound_slope = (emax - bound_red_max) / t_final;
            slope_acc = (bisect_ub_acc - bisect_lb_acc) /
                        (bisect_ub_time_step_acc - bisect_lb_time_step_acc);
            if (slope_acc < bound_slope) {
                pred_eacc = inf;
            } else {
                const double add =
                    (bound_slope * (t + bisect_lb_time_step_acc) - eacc_total - bisect_lb_acc) /
                    (slope_acc - bound_slope);
                pred_eacc = bisect_lb_time_step_acc + add;
            }
        } else {
            // Bisection: the first step halves; later ones assume a quadratic decrease.
            const double factor = k == 0 ? 0.5
                                         : std::sqrt((0.95 - bisect_lb_acc_perc) /
                                                     (bisect_ub_acc_perc - bisect_lb_acc_perc));
            pred_eacc = bisect_lb_time_step_acc +
                        factor * (bisect_ub_time_step_acc - bisect_lb_time_step_acc);
        }
    }

    // 2. Non-accumulating error.
    double pred_enonacc = inf;
    if (fullcomp) {
        if (bisect_ub_nonaccok) {
            const double slope_nonacc = (bisect_ub_nonacc - bisect_lb_nonacc) /
                                        (bisect_ub_time_step_nonacc - bisect_lb_time_step_nonacc);
            double add;
            if (is_u)
                add = (emax - bound_red_max * t / t_final - eacc_total - bisect_lb_acc -
                       bisect_lb_nonacc) /
                      (slope_acc + slope_nonacc - bound_red_max / t_final);
            else
                add = (emax - bisect_lb_nonacc) / slope_nonacc;
            pred_enonacc = bisect_lb_time_step_nonacc + add;
        } else {
            const double factor = k == 0 ? 0.5
                                         : (0.95 - bisect_lb_nonacc_perc) /
                                               (bisect_ub_nonacc_perc - bisect_lb_nonacc_perc);
            pred_enonacc = bisect_lb_time_step_nonacc +
                           factor * (bisect_ub_time_step_nonacc - bisect_lb_time_step_nonacc);
        }
        // For safety if the slopes misbehave.
        if (pred_enonacc < 0) pred_enonacc = inf;
    }
    return aux_min({pred_enonacc, pred_eacc, max_time_step});
}

double LinErrorBound::estimate_time_step_size(double t, int k, int kIter, bool fullcomp,
                                              double time_step, double max_time_step, bool is_u) {
    if (use_approx_fun) {
        time_step = approx_fun(t, k, kIter, fullcomp, time_step, max_time_step);
        // The proposal is refused if it lies below a lower bracket, above an upper bracket that
        // failed, or below an upper bracket that held.
        const double lbs[2] = {bisect_lb_time_step_acc, bisect_lb_time_step_nonacc};
        const double ubs[2] = {bisect_ub_time_step_acc, bisect_ub_time_step_nonacc};
        const bool oks[2] = {bisect_ub_accok, bisect_ub_nonaccok};
        bool refuse = false;
        for (int i = 0; i < 2; ++i)
            refuse = refuse || time_step <= lbs[i] || (time_step >= ubs[i] && !oks[i]) ||
                     (time_step <= ubs[i] && oks[i]);
        use_approx_fun = !refuse;
    }
    // The approximation functions may have been dropped above.
    if (!use_approx_fun) time_step = bisection(t, k, fullcomp, time_step, max_time_step, is_u);
    return time_step;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
