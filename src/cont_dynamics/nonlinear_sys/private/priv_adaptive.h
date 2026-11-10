// priv_adaptive - the private functions of NonlinearSys reach with alg "lin-adaptive"
//
// Not for callers of the class: reach uses them. The algorithm of Wetzlinger et al. [1] chooses
// the step size, the order of the Taylor series and the tensor order of every step itself; this
// is the state it carries from step to step (CORA's options) and the functions that update it.
//
// Syntax:   ReachSet R = priv_reach_adaptive(sys, problem, alg);
// Inputs:   see the declarations below
// Outputs:  R - the sets over and at the end of the steps, and the step times
// References:
//    [1] M. Wetzlinger, A. Kulmburg, A. Le Penven and M. Althoff. "Adaptive Reachability
//        Algorithms for Nonlinear Systems Using Abstraction Error Analysis", NAHS, 2022.
// See also: priv.h, reach.cpp
//
// CORA.cpp only: the declarations of the private functions of the adaptive algorithm.
// Authors:       Claude Sonnet 5.5
// Written:       10-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cont_dynamics/nonlinear_sys/private/priv.h"

#include <optional>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// The state of the analysis -----------------------------------------------------------------------

/// A value that CORA stores as x(i) for the step i, of which the algorithm reads the one of the
/// step and the one of the step before: `has(i)` is MATLAB's length(x) == i.
template <class T>
class PerStep {
  public:
    bool has(int i) const { return last_ == i; }
    /// The value of step i, or of the step before.
    const T &at(int i) const { return last_ == i ? value_ : before_; }
    void set(int i, T value) {
        if (last_ != i) {
            before_ = std::move(value_);
            last_ = i;
        }
        value_ = std::move(value);
    }
    void clear() { last_ = 0; }

  private:
    int last_ = 0;
    T value_{}, before_{};
};

/// The generator indices that reduce "adaptive" chose in a step, for reduce "idx" in its passes.
using IndexCache = PerStep<std::vector<int64_t>>;

/// What the algorithm carries from step to step (CORA's options and the parts of params it
/// uses); linReach changes it, reach starts it.
struct AdaptiveRun {
    alg::LinearizationAdaptive alg;
    double t_final = 0;
    std::optional<Zonotope> U;  ///< the input set as given (none: no inputs)
    std::optional<Zonotope> U0; ///< U without its center (CORA's params.U)

    int i = 1;            ///< the step
    double t = 0;         ///< its start time
    double time_step = 0; ///< its step size
    int pass = 0;         ///< CORA's options.run: 1 at the horizon, 2 at the step size
    int tensor_order = 2;
    Tensor error_adm_horizon,
        error_adm_delta_t_opt; ///< the admissible errors the next pass starts at

    PerStep<double> finite_horizon, varphi, step_size;
    PerStep<int> tt_err, tt_lin; ///< the terms of the series of the error and of e^{A dt}
    IndexCache idx_hom_ti, idx_hom_tp, idx_par, idx_red, idx_err;

    // shortcuts in the abstraction error
    bool third_order_empty = false; ///< the third derivatives are all zero
    bool hessian_const = false;     ///< the Hessian is the same over every set: hessian_value
    bool hessian_checked = false;
    std::vector<std::optional<Range>> hessian_value;

    // the powers of A that matter (CORA's aux_getPowers)
    std::optional<std::vector<char>> zero_width;
    std::vector<double> orders;
    int min_order = 0;
    double kappa_delta_t = 0, kappa_abstrerr = 0; ///< the step size and error of the last switch
};

// The functions -----------------------------------------------------------------------------------

/// The sets of a step.
struct AdaptiveStep {
    Zonotope time_interval, time_point;
};

/// reach for alg "lin-adaptive" (CORA's reach_adaptive).
ReachSet priv_reach_adaptive(const NonlinearSys &sys, const ReachProblem &problem,
                             const alg::LinearizationAdaptive &alg);

/// One step from Rstart whose size the algorithm chooses (CORA's linReach_adaptive).
AdaptiveStep priv_lin_reach_adaptive(const NonlinearSys &sys, const Zonotope &Rstart,
                                     AdaptiveRun &run);

/// The sets of the linearized system over a step and at its end (CORA's linearSys
/// initReach_adaptive); none if the series of e^{A dt} does not converge (the step is too large).
struct AdaptiveLinear {
    Zonotope time_interval, time_point;
};
std::optional<AdaptiveLinear> priv_init_reach_adaptive(const Linearized &lin,
                                                       const Zonotope &Rstart, AdaptiveRun &run);

/// The linearization error over R, a set relative to x_star, at run.tensor_order (CORA's
/// priv_abstractionError_adaptive): the set of the error and its bound per state (n, 1).
AbstractionError priv_abstrerr_adaptive(const NonlinearSys &sys, const Zonotope &R,
                                        const Tensor &x_star, const std::optional<Tensor> &u_star,
                                        AdaptiveRun &run);

/// The sum of the lengths of the generators per coordinate, the norm of a vector, and the entries
/// of a vector at the dimensions of the minimal order (CORA's orders == minorder).
std::vector<double> priv_row_sums(const Zonotope &Z);
double priv_vector_norm(const std::vector<double> &v);
std::vector<double> priv_at_min_order(const std::vector<double> &v, const AdaptiveRun &run);

/// The step size gain zeta_phi of the minimal order (CORA's zetaphi(minorder + 1)).
double priv_zeta_phi(const AdaptiveRun &run);

/// The product of the gains over the scalings from step size `prev` to `curr` by mu (CORA's
/// aux_varphitotal), each between varphi and the limit.
double priv_varphitotal(double varphi, double varphi_lim, double prev, double curr, double mu);

/// The step size below the horizon that minimizes the estimated width of the sets over the
/// following steps (CORA's aux_optimaldeltat, eq. (49) of [1]).
double priv_optimal_delta_t(const Zonotope &Rt, const Zonotope &Rerr, double deltat,
                            double varphimin, double zeta_p, const AdaptiveRun &run);

/// The gain that makes the error at the step size deltat that of the horizon, by bisection
/// (CORA's aux_varphiest).
double priv_varphi_est(double horizon, double deltat, const Zonotope &Rerr_h,
                       const Zonotope &Rerr_deltat, const AdaptiveRun &run);

/// reduce "adaptive" at diag_percent, or "idx" with the indices the cache holds for the step.
Zonotope priv_reduce_cached(const Zonotope &Z, double diag_percent, IndexCache &cache, int step);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
