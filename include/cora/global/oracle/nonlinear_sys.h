// nonlinear_sys - lean::NonlinearSys, lin-adaptive reachability of x' = f(x, u) by CORALean
//
// Syntax:     lean::NonlinearSys sys(my_sys);   auto run = sys.reach_adaptive(R0, U, t_final, alg);
//             double v = lean::NonlinearSys::varphi_total_adaptive(alg, varphi, lim, prev, curr);
// Operations: reach_adaptive and one method per `nonlinearSys.*Adaptive` op of the oracle:
//             lin_reach_adaptive_pass, abstr_err_adaptive, orders_adaptive, and the three tuning
//             formulas optimal_delta_t_adaptive, varphi_total_adaptive, varphi_est_adaptive
// Outputs:    the oracle's float definitions at a dtype of the unguaranteed ops (nearest)
// Scheme:     the dynamics go as the tree of the system's expressions; CORALean differentiates it
// See also:   lean/linear_sys.h, contDynamics/nonlinear_sys/nonlinear_sys.h
//
// Authors:       Claude Sonnet 5.5
// Written:       10-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_dynamics/algorithms.h"
#include "cora/cont_dynamics/nonlinear_sys/nonlinear_sys.h"
#include "cora/global/oracle/request.h"

#include <optional>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::lean {

/// The run of reach_adaptive: the step sizes, the tensor order of every step, the time reached, the
/// set at the end and over every step.
struct AdaptiveRun {
    std::vector<double> steps;
    std::vector<int64_t> tensor_orders;
    double t;
    Zonotope time_point;
    std::vector<Zonotope> time_interval;
};

/// One pass of the error loop: the sets and what the tuning reads of them.
struct AdaptivePassResult {
    Zonotope time_interval, time_point, Rmax;
    cora::Tensor A, xs, true_error, abstrerr;
    std::vector<bool> g_zero;
};

/// VerrorDyn and its true error (the largest magnitude of its box).
struct AbstrErrResult {
    Zonotope V;
    cora::Tensor true_error;
};

class NonlinearSys {
  public:
    /// The system of sys's expressions.
    explicit NonlinearSys(const cora::NonlinearSys &sys);

    /// CORA's lin-adaptive from R0 with the input set U (none: no input) to t_final.
    AdaptiveRun reach_adaptive(const Zonotope &R0, const std::optional<Zonotope> &U, double t_final,
                               const alg::LinearizationAdaptive &alg) const;

    /// One pass at time_step from R with the tensor order 2 or 3 and the admissible error.
    std::optional<AdaptivePassResult>
    lin_reach_adaptive_pass(const std::optional<Zonotope> &U, const Zonotope &R, double time_step,
                            int tensor_order, const cora::Tensor &error_adm,
                            const alg::LinearizationAdaptive &alg) const;

    /// VerrorDyn for Rmax around the linearization point xs.
    AbstrErrResult abstr_err_adaptive(const std::optional<Zonotope> &U, const cora::Tensor &xs,
                                      const Zonotope &Rmax, int tensor_order,
                                      double red_factor) const;

    /// The orders q_i of the abstraction error (none: Inf).
    std::vector<std::optional<int64_t>>
    orders_adaptive(const std::optional<Zonotope> &U, const cora::Tensor &A, const cora::Tensor &xs,
                    const std::vector<bool> &zero_width, const std::vector<bool> &g_zero) const;

    /// The tuning formulas, which need no dynamics.
    static double optimal_delta_t_adaptive(const alg::LinearizationAdaptive &alg, double red_factor,
                                           double r_radius, double rerr1, double deltat,
                                           double varphimin, double zeta_p);
    static double varphi_total_adaptive(const alg::LinearizationAdaptive &alg, double varphi,
                                        double varphi_lim, double prev, double curr);
    static double varphi_est_adaptive(const alg::LinearizationAdaptive &alg,
                                      const cora::Tensor &abstrerr_h, const cora::Tensor &abstrerr,
                                      double horizon, double deltat, const std::vector<bool> &mask,
                                      int minorder);

  private:
    /// The members every op on the dynamics reads: n, m, f and p.
    std::vector<Operand> system() const;

    int64_t n_, m_;
    Json f_;
};

} // namespace cora::lean

// ---------------------------------------  END OF CODE  ---------------------------------------- //
