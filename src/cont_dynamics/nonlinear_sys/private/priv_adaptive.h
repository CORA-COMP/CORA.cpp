// priv_adaptive - the private functions of NonlinearSys reach with alg "lin-adaptive"
//
// Not for callers of the class: reach uses them. CORA's algorithm of Wetzlinger et al. [1], as
// CORALean repairs it (reachAdaptiveFloat): the Hessian is read over the box of Rmax joined with
// the linearization point, the tensor order 3 is the Hessian at the point on the reduced set plus
// the difference of the Hessian over the box, and a window is one linear step with the error as
// input. This is the state it carries from step to step and the functions that update it; the sums
// add from the last term to the first and a norm is the root of the sum of squares, as CORALean's
// sumUp and normUp at nearest.
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
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_dynamics/nonlinear_sys/nonlinear_sys.h"

#include <optional>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// The state ---------------------------------------------------------------------------------------

using Vec = std::vector<double>;

/// What a step hands the next (CORALean's LinReachAdaptiveState, CORA's options): the number and
/// the start time of the next step, the sets, the tuning and the orders of the abstraction error
/// (-1 for Inf).
struct AdaptiveState {
    /// The state before the first step (linReachAdaptiveInit): at t0 from R0 in n dimensions.
    AdaptiveState(const Zonotope &R0, int64_t n, double t0)
        : t(t0), R(R0), Rti(R0), err_adm_horizon(std::size_t(n), 0.0),
          err_adm_deltatopt(std::size_t(n), 0.0), orders(std::size_t(n), -1),
          zero_width(std::size_t(n), 0) {}

    int i = 1;
    double t = 0;
    Zonotope R, Rti;
    double time_step = 0, finitehorizon = 0, varphi = 0;
    Vec err_adm_horizon, err_adm_deltatopt;
    int tensor_order = 2;
    std::vector<int> orders;
    int minorder = 0;
    std::vector<char> zero_width;
    double kappa_deltat = 0, kappa_abstrerr = 0;
    Vec steps; ///< the step sizes, latest first
};

/// One linearization at a fixed step size: the sets with the error and what the tuning reads of it.
struct AdaptivePass {
    Zonotope Rti, Rtp;
    Tensor A;
    Vec xs;
    Zonotope Rmax;
    Vec true_error, abstrerr;
    std::vector<char> gzero;
};

// The boxes (priv_abstrerr_adaptive.cpp) ---------------------------------------------------------

/// A box as the bounds of its coordinates.
struct Box {
    Vec lo, hi;
};

/// The values as a column (n, 1) like the tensor.
Tensor priv_column(const Tensor &like, const Vec &v);

/// The box of a zonotope, c -/+ the sum of the absolute generators (CORALean's Zonotope.interval).
Box priv_zonotope_box(const Zonotope &Z);

/// The zonotope of a box: its center, and one diagonal generator per coordinate
/// (Interval.zonotope).
Zonotope priv_box_zonotope(const Tensor &like, const Box &b);

/// f over the box x and u (none: no inputs).
Box priv_enclosure(const NonlinearSys &sys, const Tensor &like, const Box &x,
                   const std::optional<Box> &u);

/// The sum of the absolute generators per coordinate (the radius of the box of Z).
Vec priv_row_sums(const Zonotope &Z);

/// The largest magnitude of each coordinate of the box (CORALean's Interval.normMag).
Vec priv_norm_mag(const Box &b);

// The tuning (priv_tune_adaptive.cpp) -------------------------------------------------------------

/// The sum from the last entry to the first, and the Euclidean norm of the magnitudes.
double priv_sum_up(const Vec &v);
double priv_norm_up(const Vec &v);

/// x^k by k multiplications (CORALean's powUp).
double priv_pow(double x, int k);

/// aux_varphitotal: the product of the interpolated gains from step size curr to prev.
double priv_varphi_total(double varphi, double varphi_lim, double prev, double curr, double decr);

/// aux_varphiest: the gain whose total scaling from deltat to horizon matches the errors.
double priv_varphi_est(double horizon, double deltat, const Vec &abstrerr_h, const Vec &abstrerr,
                       double decr, const std::vector<char> &mask, int minorder, int fuel);

/// aux_optimaldeltat at alg "lin": the candidate step size with the smallest estimated size.
double priv_optimal_delta_t(double mu, double red_factor, double r_radius, double rerr1,
                            double deltat, double varphimin, double zeta_p);

/// aux_getPowers at alg "lin": the order q_i of the abstraction error (-1 for Inf), from the
/// pattern of A, the zero-width dimensions, the zero generators of the error and the Hessian.
std::vector<int> priv_adaptive_orders(const Tensor &A, const std::vector<char> &zero_width,
                                      const std::vector<char> &gzero,
                                      const std::vector<std::optional<Range>> &H, int64_t n,
                                      int64_t m, int fuel);

// The steps ---------------------------------------------------------------------------------------

/// VerrorDyn (CORA's priv_abstractionError_adaptive) for Rmax, absolute, at tensor order 2 or 3.
Zonotope priv_abstr_err_adaptive(const NonlinearSys &sys, int kappa, const Tensor &A,
                                 const Tensor &B, const Vec &xs, const Zonotope &Rmax,
                                 const std::optional<Zonotope> &U, double red_factor);

/// One linearization at time_step from R: the error loop from error_adm, then the sets; none where
/// the loop or the linear step fails.
std::optional<AdaptivePass> priv_lin_reach_adaptive_pass(const NonlinearSys &sys, int kappa,
                                                         const std::optional<Zonotope> &U,
                                                         const alg::LinearizationAdaptive &alg,
                                                         double time_step, const Zonotope &R,
                                                         const Vec &error_adm);

/// One step of CORA's linReach_adaptive from st: the state after it, its sets unreduced.
AdaptiveState priv_lin_reach_adaptive(const NonlinearSys &sys, const std::optional<Zonotope> &U,
                                      const alg::LinearizationAdaptive &alg, double t_final,
                                      const AdaptiveState &st);

/// reach for alg "lin-adaptive".
ReachSet priv_reach_adaptive(const NonlinearSys &sys, const ReachProblem &problem,
                             const alg::LinearizationAdaptive &alg);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
