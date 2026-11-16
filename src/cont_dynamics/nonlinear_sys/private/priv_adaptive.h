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
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       11-November-2026
// Last update:   16-November-2026
// Reviewed:      ---

#pragma once

#include "cora/cont_dynamics/nonlinear_sys/nonlinear_sys.h"
#include "global/exact_time.h"

#include <algorithm>
#include <functional>
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
        : t(ExactTime(t0)), R(R0), Rti(R0), err_adm_horizon(std::size_t(n), 0.0),
          err_adm_deltatopt(std::size_t(n), 0.0), orders(std::size_t(n), -1),
          zero_width(std::size_t(n), 0) {}

    int i = 1;
    ExactTime t; ///< exact, as CORALean's rational time
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

/// The radius (the row sums of |G|) of the solution of x' = A x + v, v in V, from 0 over dt at the
/// order eta: CORALean's errorSolutionAdaptiveFloat, which the tuning reads
/// (priv_error_radius_float.cpp).
Vec priv_error_radius_float(const Tensor &A, double dt, int eta, const Zonotope &V);

/// Whether the main loop takes another step (CORALean's reachAdaptiveFloat, exactly): more than
/// kHorizonEnd of the time is left, and it is at most 10^9 times the last ten steps (CORA's
/// aux_checkForAbortion: the step sizes have not converged to 0).
inline bool priv_adaptive_continues(const Vec &steps, const ExactTime &t, double t_final) {
    const ExactTime left = ExactTime(t_final) - t;
    ExactTime last;
    for (std::size_t k = 0; k < std::min<std::size_t>(10, steps.size()); ++k)
        last = last + steps[k];
    return ExactTime(numerics::kHorizonEnd) < left &&
           (steps.empty() || !(last.times(1000000000U) < left));
}

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

/// The box of b joined with the point p (the Hessian region must contain the expansion point).
Box priv_hull(const Box &b, const Vec &p);

/// The Hessian over a box as bounds (lo, hi) of the entries (i, a, b), row-major, 0 where it is.
struct Hessian {
    Vec lo, hi;
};
Hessian priv_hessian(const NonlinearSys &sys, const Tensor &like, const Box &x,
                     const std::optional<Box> &u);

/// The box [-l, l] with l_i = 1/2 dz' max(|H_i - Hp|) dz, dz the largest deviation of X and U from
/// the expansion point (hessRemainder; Hp the Hessian to subtract, empty for none).
Box priv_hessian_remainder(const Hessian &H, const Vec &Hp, const Box &X,
                           const std::optional<Box> &U, const Vec &xs, const Vec &us, int64_t n);

// The linearization -------------------------------------------------------------------------------

/// The matrices 1/2 H_i of the Hessian at the point, (n + m, n + m) each (the quadratic maps of the
/// second-order error).
std::vector<Tensor> priv_half_hessians(const Tensor &like, const Hessian &Hp, int64_t n, int64_t d);

/// What a pass linearizes (CORA's linearize at the center of the set): the point x* = c + dt/2 f(c,
/// u*), the Jacobians there and the constant input f(x*, u*) with B (U - u*) of the linear system.
struct AdaptiveLinearization {
    Vec xs;
    Tensor xs_t;
    Jacobians J;
    Zonotope Vbase;
};
AdaptiveLinearization priv_linearize_adaptive(const NonlinearSys &sys,
                                              const std::optional<Zonotope> &U,
                                              const Tensor &center, double dt);

/// The error loop of a pass: from the admissible error, the true error of the guess (computed by
/// true_error_of, which keeps what else it computes) is checked and the guess becomes 1.1 times
/// it, until it holds; none if the ratio does not shrink after two rounds or the fuel runs out.
std::optional<Vec> priv_error_loop_adaptive(const Vec &error_adm, int fuel,
                                            const std::function<Vec(const Vec &)> &true_error_of);

// The tuning (priv_tune_adaptive.cpp) -------------------------------------------------------------

/// The sum from the last entry to the first, and the Euclidean norm of the magnitudes.
double priv_sum_up(const Vec &v);
double priv_norm_up(const Vec &v);

/// The zero-width dimensions, the orders of the abstraction error and their least one.
struct OrdersState {
    std::vector<char> zero_width;
    std::vector<int> orders;
    int minorder = 0;
};

/// The update of the orders after a pass (aux_getPowers), at the very first step or when the
/// dimensions of zero width (row sums of the generators of the set at most kZeroWidth) change.
void priv_update_orders(const NonlinearSys &sys, const std::optional<Zonotope> &U,
                        const Tensor &like, const Vec &row_sums, bool veryfirststep,
                        const Tensor &A, const Vec &xs, const std::vector<char> &gzero, int shift,
                        OrdersState &st);

/// withinTol(a, b, tol): absolutely or relatively within tol.
bool priv_within_tol(double a, double b, double tol);

/// max(errRatio(~isnan(errRatio))) over the dimensions of the mask, 0/0 skipped.
double priv_max_ratio(const std::vector<char> &mask, const Vec &num, const Vec &den);

/// zetaP: exp(trace(A dt)) from the diagonal of A.
double priv_zeta_p(const Tensor &A, double time_step);

/// x^k by k multiplications (CORALean's powUp).
double priv_pow(double x, int k);

/// aux_varphitotal: the product of the interpolated gains from step size curr to prev.
double priv_varphi_total(double varphi, double varphi_lim, double prev, double curr, double decr);

/// aux_varphiest: the gain whose total scaling from deltat to horizon matches the errors.
double priv_varphi_est(double horizon, double deltat, const Vec &abstrerr_h, const Vec &abstrerr,
                       double decr, const std::vector<char> &mask, int minorder, int fuel);

/// aux_optimaldeltat: the candidate step size with the smallest estimated size. dHused is 0.5 at
/// alg "lin" (none), 0.3 at "poly".
double priv_optimal_delta_t(double mu, double red_factor, double r_radius, double rerr1,
                            double deltat, double varphimin, double zeta_p,
                            std::optional<double> d_h_used = std::nullopt);

/// aux_getPowers: the order q_i of the abstraction error (-1 for Inf), from the pattern of A, the
/// zero-width dimensions, the zero generators of the error and the Hessian. shift is 1 at "poly"
/// when the third-order tensor is empty (sigma + 1), else 0.
std::vector<int> priv_adaptive_orders(const Tensor &A, const std::vector<char> &zero_width,
                                      const std::vector<char> &gzero,
                                      const std::vector<std::optional<Range>> &H, int64_t n,
                                      int64_t m, int fuel, int shift = 0);

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
