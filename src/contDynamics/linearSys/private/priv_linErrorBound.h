// priv_linErrorBound - the error bookkeeping of the adaptive reachability algorithm
//
// As CORA's linErrorBound.
// The adaptive algorithm keeps the Hausdorff distance of its sets to the exact ones below emax
// [1]: accumulating errors (the particular solution of the uncertain input), a reduction error
// and non-accumulating errors of each step; this class holds them, their bounds over time, and
// predicts the time step size that meets the bounds (by approximation functions or bisection).
// Indices k (step) and kIter (trial of a step size within the step) are 0-based here.
//
// Syntax:   LinErrorBound errs(emax, tFinal);   errs.nextBounds(timeStep, t, k, kIter);
// Inputs:   emax - error margin;  tFinal - time horizon
// Outputs:  the bounds and checks below
// References:
//    [1] M. Wetzlinger et al. "Fully automated verification of linear systems using inner-
//        and outer-approximations of reachable sets", TAC, 2023.
// See also: priv_reach_adaptive

#pragma once

#include "contSet/contSet.h"

#include <limits>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

class LinErrorBound {
  public:
    using PerStep = std::vector<std::vector<double>>;  ///< [k][kIter]

    LinErrorBound(double emax, double tFinal);

    // Errors and bounds ---------------------------------------------------------------------------

    double emax, tFinal;
    std::vector<std::vector<double>> timeSteps;
    PerStep step_acc, seq_nonacc, bound_acc, bound_rem, bound_red;
    std::vector<double> step_red, cum_acc, cum_red, bound_remacc;
    double bound_red_max = 0;
    std::vector<std::vector<char>> bound_acc_ok, bound_nonacc_ok;
    PerStep idv_PUtkplus1, idv_F, idv_G, idv_linComb, idv_PUtauk;
    bool useApproxFun = false;

    // Approximation functions (a dt^2 + b dt or b dt) ---------------------------------------------

    double coeff_PUtkplus1_a = 0, coeff_PUtkplus1_b = 0, coeff_F_a = 0, coeff_F_b = 0;
    double coeff_G_a = 0, coeff_G_b = 0, coeff_linComb_b = 0, coeff_PUtauk_b = 0;

    // Bisection: the bracket of the step size per error class -------------------------------------

    double bisect_lb_timeStep_acc = 0, bisect_lb_acc = 0, bisect_lb_acc_perc = 0;
    double bisect_lb_timeStep_nonacc = 0, bisect_lb_nonacc = 0, bisect_lb_nonacc_perc = 0;
    double bisect_ub_timeStep_acc = 0, bisect_ub_acc = 0, bisect_ub_acc_perc = 0;
    double bisect_ub_timeStep_nonacc = 0, bisect_ub_nonacc = 0, bisect_ub_nonacc_perc = 0;
    bool bisect_lb_accok = true, bisect_lb_nonaccok = true;
    bool bisect_ub_accok = true, bisect_ub_nonaccok = true;

    // Operations ----------------------------------------------------------------------------------

    /// The error bounds of step k for the trial step size timeStep at time t.
    void nextBounds(double timeStep, double t, int k, int kIter);

    /// Commits the errors of trial kIter of step k to the running sums.
    void accumulateErrors(int k, int kIter);

    /// Keeps only the values of the chosen trial in every per-step list.
    void removeRedundantValues(int k, int kIter);

    /// The errors of the time-interval and time-point sets of step k.
    void fullErrors(int k, double &RcontError, double &RcontTpError) const;

    /// Whether the errors respect their bounds (false: a bug); internalOk is false if the
    /// bookkeeping of the bounds is inconsistent.
    bool checkErrors(const std::vector<double> &RoutError, const std::vector<double> &RoutTpError,
                     bool &internalOk) const;

    /// The share bound_red_max of emax that the reduction error may take, by the heuristic of
    /// [1, Sec. IV.D.2], from the system matrix A (n, n) and the input generators GU (n, m).
    void computeErrorBoundReduction(const Tensor &A, const Tensor &GU);

    /// Coefficients of the approximation functions from the errors of the last two trials.
    void updateCoefficientsApproxFun(int k, int kIter, bool fullcomp);

    /// The bracket of the step size, updated with trial kIter.
    void updateBisection(int k, int kIter, bool isU, double timeStep);

    /// The next trial step size; the approximation functions are dropped for bisection if they
    /// propose a value outside the bracket.
    double estimateTimeStepSize(double t, int k, int kIter, bool fullcomp, double timeStep,
                                double maxTimeStep, bool isU);

  private:
    double approxFun(double t, int k, int kIter, bool fullcomp, double timeStep,
                     double maxTimeStep) const;
    double bisection(double t, int k, bool fullcomp, double timeStep, double maxTimeStep,
                     bool isU) const;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
