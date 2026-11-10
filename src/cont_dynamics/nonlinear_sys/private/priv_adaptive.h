// priv_adaptive - the private functions of NonlinearSys reach with alg "lin-adaptive"
//
// Not for callers of the class: reach uses them. The algorithm of Wetzlinger et al. [1] chooses
// the size of every step and the tensor order of its abstraction error itself; this is the state
// it carries from step to step and the functions that update it, as CORALean's adaptStepFloat
// defines them (the step, the search of the step size, the gain and the order).
//
// Syntax:   ReachSet R = priv_reach_adaptive(sys, problem, alg);
// Inputs:   see the declarations below
// Outputs:  R - the sets over and at the end of the steps, and the step times
// References:
//    [1] M. Wetzlinger, A. Kulmburg, A. Le Penven and M. Althoff. "Adaptive Reachability
//        Algorithms for Nonlinear Systems Using Abstraction Error Analysis", NAHS, 2022.
// See also: reach.cpp
//
// CORA.cpp only: the declarations of the private functions of the adaptive algorithm.
// Authors:       Claude Sonnet 5.5
// Written:       10-November-2026
// Last update:   10-November-2026
// Reviewed:      ---

#pragma once

#include "cora/cont_dynamics/nonlinear_sys/nonlinear_sys.h"

#include <memory>
#include <optional>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// The state ---------------------------------------------------------------------------------------

using Vec = std::vector<double>;

/// What a step carries to the next (CORALean's AdaptState): the time, the set at that time and
/// the set over the step before it, the finite horizon, the gain, the tensor order (1 or 2) and
/// the error radius of the step before.
struct AdaptState {
    double t = 0;
    Zonotope R, Rti;
    double h = 0, phi = 0;
    int kappa = 1;
    double rabs = 0;
};

/// The sets over a step and at its end.
struct StepSets {
    Zonotope time_interval, time_point;
};

/// The linearized system's matrix A and the sets it reaches from R (adaptLinPassFloat).
struct LinPass {
    Tensor A;
    StepSets sets;
};

/// The step size the search chose, Δt* = h ζ_δ^exponent.
struct StepSize {
    double time_step;
    int exponent;
};

// The tuning --------------------------------------------------------------------------------------

/// x^k by k multiplications (CORALean's powUp), the order that rounding sees.
double priv_pow(double x, int k);

/// exp(a) from above and from below by squaring a seed at a / 2^depth, depth times.
double priv_exp_up(int depth, double a);
double priv_exp_down(int depth, double a);

/// log(a) from above by bisection on [0, a] with `steps` halvings (CORALean's logUp).
double priv_log_up(int depth, int steps, double a);

/// phi(j + 1) of eq. (54): the gain between phi1 at the horizon and the limit zeta_delta.
double priv_interp_gain(double zeta_delta, double phi1, int j);

/// Eq. (58): the step size h zeta_delta^s for the s that estimates the smallest set (eq. (57)),
/// from the width of R, the error radius at the horizon and zeta_a = e^{tr(A) h}.
StepSize priv_opt_delta_t(const alg::LinearizationAdaptive &alg, double zeta_a, double phi,
                          double r0, double rabs, double h);

/// Eq. (56): the gain phi1 in [0, zeta_delta] that explains the error radius found at the step
/// size.
double priv_estimate_gain(double zeta_delta, double lam, double rabs1, double rabs_opt,
                          int scalings, int fuel);

/// Line 13 of Alg. 2, eqs. (46) and (47): the tensor order of the next step.
int priv_tune_order(double zeta_k, double phi, double rabs, double rabs_last, const Vec &d1,
                    const Vec &d2, int kappa);

// The steps ---------------------------------------------------------------------------------------

/// The passes of one step size dt from the set R: the linear system's sets without error, and the
/// sets of a step at a tensor order (1: slopes, 2: Hessian). They share the expansion point, the
/// linearization and its powers; order_step throws where the flow does not stay in the region.
class AdaptPass {
  public:
    AdaptPass(const NonlinearSys &sys, const std::optional<Zonotope> &U,
              const alg::LinearizationAdaptive &alg, double dt, const Zonotope &R);
    LinPass lin_pass();
    StepSets order_step(int kappa);

  private:
    struct Impl;
    std::shared_ptr<Impl> impl_;
};

/// Alg. 2: one adaptively tuned step.
AdaptState priv_adapt_step(const NonlinearSys &sys, const std::optional<Zonotope> &U,
                           const alg::LinearizationAdaptive &alg, const AdaptState &s);

/// reach for alg "lin-adaptive".
ReachSet priv_reach_adaptive(const NonlinearSys &sys, const ReachProblem &problem,
                             const alg::LinearizationAdaptive &alg);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
