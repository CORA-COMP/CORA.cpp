// priv_lin_error_bound - the error bookkeeping of the adaptive reachability algorithm
//
// As CORA's linErrorBound.
// The adaptive algorithm keeps the Hausdorff distance of its sets to the exact ones below emax
// [1]: accumulating errors (the particular solution of the uncertain input), a reduction error
// and non-accumulating errors of each step; this class holds them, their bounds over time, and
// predicts the time step size that meets the bounds (by approximation functions or bisection).
// Indices k (step) and kIter (trial of a step size within the step) are 0-based here.
//
// Syntax:   LinErrorBound errs(emax, t_final);   errs.next_bounds(time_step, t, k, kIter);
// Inputs:   emax - error margin;  t_final - time horizon
// Outputs:  the bounds and checks below
// References:
//    [1] M. Wetzlinger et al. "Fully automated verification of linear systems using inner-
//        and outer-approximations of reachable sets", TAC, 2023.
// See also: priv_reach_adaptive

#pragma once

#include "cora/cont_set/cont_set.h"

#include <limits>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

class LinErrorBound {
  public:
    using PerStep = std::vector<std::vector<double>>; ///< [k][kIter]

    LinErrorBound(double emax, double t_final);

    // Errors and bounds ---------------------------------------------------------------------------

    double emax, t_final;
    std::vector<std::vector<double>> time_steps;
    PerStep step_acc, seq_nonacc, bound_acc, bound_rem, bound_red;
    std::vector<double> step_red, cum_acc, cum_red, bound_remacc;
    double bound_red_max = 0;
    std::vector<std::vector<char>> bound_acc_ok, bound_nonacc_ok;
    PerStep idv_pu_tkplus1, idv_f, idv_g, idv_lin_comb, idv_pu_tauk;
    bool use_approx_fun = false;

    // Approximation functions (a dt^2 + b dt or b dt) ---------------------------------------------

    double coeff_pu_tkplus1_a = 0, coeff_pu_tkplus1_b = 0, coeff_f_a = 0, coeff_f_b = 0;
    double coeff_g_a = 0, coeff_g_b = 0, coeff_lin_comb_b = 0, coeff_pu_tauk_b = 0;

    // Bisection: the bracket of the step size per error class -------------------------------------

    double bisect_lb_time_step_acc = 0, bisect_lb_acc = 0, bisect_lb_acc_perc = 0;
    double bisect_lb_time_step_nonacc = 0, bisect_lb_nonacc = 0, bisect_lb_nonacc_perc = 0;
    double bisect_ub_time_step_acc = 0, bisect_ub_acc = 0, bisect_ub_acc_perc = 0;
    double bisect_ub_time_step_nonacc = 0, bisect_ub_nonacc = 0, bisect_ub_nonacc_perc = 0;
    bool bisect_lb_accok = true, bisect_lb_nonaccok = true;
    bool bisect_ub_accok = true, bisect_ub_nonaccok = true;

    // Operations ----------------------------------------------------------------------------------

    /// The error bounds of step k for the trial step size time_step at time t.
    void next_bounds(double time_step, double t, int k, int kIter);

    /// Commits the errors of trial kIter of step k to the running sums.
    void accumulate_errors(int k, int kIter);

    /// Keeps only the values of the chosen trial in every per-step list.
    void remove_redundant_values(int k, int kIter);

    /// The errors of the time-interval and time-point sets of step k.
    void full_errors(int k, double &RcontError, double &RcontTpError) const;

    /// Whether the errors respect their bounds (false: a bug); internal_ok is false if the
    /// bookkeeping of the bounds is inconsistent.
    bool check_errors(const std::vector<double> &RoutError, const std::vector<double> &RoutTpError,
                      bool &internal_ok) const;

    /// The share bound_red_max of emax that the reduction error may take, by the heuristic of
    /// [1, Sec. IV.D.2], from the system matrix A (n, n) and the input generators GU (n, m).
    void compute_error_bound_reduction(const Tensor &A, const Tensor &GU);

    /// Coefficients of the approximation functions from the errors of the last two trials.
    void update_coefficients_approx_fun(int k, int kIter, bool fullcomp);

    /// The bracket of the step size, updated with trial kIter.
    void update_bisection(int k, int kIter, bool is_u, double time_step);

    /// The next trial step size; the approximation functions are dropped for bisection if they
    /// propose a value outside the bracket.
    double estimate_time_step_size(double t, int k, int kIter, bool fullcomp, double time_step,
                                   double max_time_step, bool is_u);

  private:
    double approx_fun(double t, int k, int kIter, bool fullcomp, double time_step,
                      double max_time_step) const;
    double bisection(double t, int k, bool fullcomp, double time_step, double max_time_step,
                     bool is_u) const;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
