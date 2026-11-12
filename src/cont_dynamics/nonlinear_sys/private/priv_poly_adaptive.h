// priv_poly_adaptive - the private functions of NonlinearSys reach with alg "poly-adaptive"
//
// Not for callers of the class: reach uses them. CORA's algorithm of Wetzlinger et al. [1] on
// polynomial zonotopes, as CORALean repairs it (reachPolyAdaptiveFloat): the Hessian is read over
// the box of Rmax joined with the linearization point, the errors RallError and VerrorDyn are
// inputs of a window over the whole step, the static error Vstat enters the end set through its
// dependent part and the input through its independent one, the second-order error is the three
// products of the start set and the difference, a set is restructured when its independent part is
// too large, and every reduction boxes an even-exponent generator at its full radius. This is the
// state the steps carry and the functions that update it.
//
// Syntax:   PolyReachSet R = priv_reach_poly_adaptive(sys, problem, alg);
// Inputs:   see the declarations below
// Outputs:  R - the sets over and at the end of the steps, and the step times
// References:
//    [1] M. Wetzlinger, A. Kulmburg, A. Le Penven and M. Althoff. "Adaptive Reachability
//        Algorithms for Nonlinear Systems Using Abstraction Error Analysis", NAHS, 2022.
// See also: priv_adaptive.h, reach.cpp
//
// CORA.cpp only: the declarations of the private functions of the poly-adaptive algorithm.
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cont_dynamics/linear_sys/private/priv_steps.h"
#include "cont_dynamics/nonlinear_sys/private/priv_adaptive.h"

#include <optional>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// The state ---------------------------------------------------------------------------------------

/// What a step hands the next (CORALean's LinReachPolyAdaptiveState, CORA's options): the step
/// number and start time, the sets, the tuning and the orders of the error (-1 for Inf).
struct PolyAdaptiveState {
    /// The state before the first step (linReachPolyAdaptiveInit): at t0 from R0.
    PolyAdaptiveState(const PolyZonotope &R0, int64_t n, double t0)
        : t(t0), R(R0), Rti(R0), err_adm_horizon(std::size_t(n), 0.0),
          err_adm_deltatopt(std::size_t(n), 0.0), orders(std::size_t(n), -1),
          zero_width(std::size_t(n), 0) {}

    int i = 1;
    double t = 0;
    PolyZonotope R, Rti;
    double time_step = 0, finitehorizon = 0, varphi = 0;
    Vec err_adm_horizon, err_adm_deltatopt;
    std::vector<int> orders;
    int minorder = 0;
    std::vector<char> zero_width;
    Vec steps; ///< the step sizes, latest first
};

/// One linearization at a fixed step size: the sets with the error and what the tuning reads of it.
struct PolyAdaptivePass {
    PolyZonotope Rti, Rtp;
    Tensor A;
    Vec xs;
    Vec true_error, abstrerr;
    std::vector<char> gzero;
};

/// The static error of a step (CORA's priv_precompStatError_adaptive): Vstat as a polynomial
/// zonotope over the factors of the start set, and Zdelta, the zonotope of the start set with the
/// input.
struct PolyStatic {
    PolyZonotope Vstat;
    Zonotope Zdelta;
};

// The sets (priv_poly_ops.cpp) --------------------------------------------------------------------

/// CORA's reduce(pZ, "adaptive", diagpercent), with CORALean's repair: the generators the rule
/// ranks first are boxed into the independent part, an even-exponent one at its full radius.
PolyZonotope priv_reduce_adaptive_poly(const PolyZonotope &Z, double diagpercent);

/// CORA's aux_reduceOnlyDep: past (order + 1) n dependent generators, all but the order * n longest
/// are boxed into the independent part.
PolyZonotope priv_reduce_only_dep(const PolyZonotope &Z, int order);

/// CORALean's restructure: the independent part is boxed into one generator per coordinate, and
/// those are made dependent (each on a factor of its own).
PolyZonotope priv_restructure(const PolyZonotope &Z);

/// Whether the independent part is too large against the dependent one (approxVolumeRatio with the
/// interval method above ratio): the product of the radii of the box of the independent part
/// exceeds ratio^n times that of the box of the dependent part.
bool priv_volume_ratio_exceeds(const PolyZonotope &Z, double ratio);

/// The generators of the zonotope of Z (even exponents halved), the radius of its box per
/// coordinate.
Vec priv_poly_radius(const PolyZonotope &Z);

/// Z with each independent generator made a dependent one on a factor of its own (CORALean's
/// absorbRounding), so that a quadratic map keeps their products. The identifiers of the new
/// factors are above those of Z and above `above` (those of the sets it is later added to).
PolyZonotope priv_absorb_independent(const PolyZonotope &Z, int64_t above = 0);

// The errors and the linear steps -----------------------------------------------------------------

/// priv_precompStatError_adaptive for the start set Rdelta (relative to xs) and the input set U.
PolyStatic priv_precomp_stat_error_adaptive(const NonlinearSys &sys, const Vec &xs,
                                            const std::optional<Zonotope> &U, double red_factor,
                                            const PolyZonotope &Rdelta);

/// VerrorDyn of priv_abstractionError_adaptive at "poly", tensor order 3: the three second-order
/// products of Zdelta and the differences Rdiff, and the Hessian remainder over the box of Rmax.
Zonotope priv_abstr_err_poly_adaptive(const NonlinearSys &sys, const Vec &xs, const Zonotope &Rmax,
                                      const Zonotope &Zdelta, const PolyZonotope &Rdiff,
                                      const std::optional<Zonotope> &U, double red_factor);

/// The integral of e^{As} over the step as an interval matrix (CORA's eAtInt of
/// errorSolution_adaptive): the series up to eta and the remainder dt [-W, W].
Interval priv_adaptive_integral(const LinearSys &lin, TaylorLinSys &taylor, double dt, int eta);

/// The window over a step and the differences to its start of x' = A x + u, u in V, from X0 (CORA's
/// initReach_adaptive with a polynomial zonotope): V is time-varying, its center held through
/// hull(0, u). The Taylor order eta must be admissible.
std::pair<PolyZonotope, PolyZonotope> priv_poly_window(const LinearSys &lin, TaylorLinSys &taylor,
                                                       double dt, int eta, const Zonotope &V,
                                                       const PolyZonotope &X0, double zeta_z);

/// The end set of the step with the constant input Vs (a polynomial zonotope over the factors of
/// X0): its dependent part through the integral of e^{As} and exact_plus, its independent part in
/// V.
PolyZonotope priv_poly_point(const LinearSys &lin, TaylorLinSys &taylor, double dt, int eta,
                             const Zonotope &V, const PolyZonotope &X0, const PolyZonotope &Vs,
                             double zeta_z);

/// One linearization at time_step from R: the error loop from error_adm, then the sets; none where
/// the loop or the linear step fails.
std::optional<PolyAdaptivePass>
priv_lin_reach_poly_adaptive_pass(const NonlinearSys &sys, const std::optional<Zonotope> &U,
                                  const alg::PolynomializationAdaptive &alg, double time_step,
                                  const PolyZonotope &R, const Vec &error_adm);

/// The Taylor order of a window: the least order from the given one (at least 2) with eta + 2 >=
/// 2 ||A dt||, none if there is none up to 4096 (CORALean's reachOrderFloat).
std::optional<int> priv_reach_order(const Tensor &A, double dt, int eta);

/// One step of CORA's linReach_adaptive at "poly" from st: the state after it, its sets unreduced.
PolyAdaptiveState priv_lin_reach_poly_adaptive(const NonlinearSys &sys,
                                               const std::optional<Zonotope> &U,
                                               const alg::PolynomializationAdaptive &alg,
                                               bool third_order_empty, double t_final,
                                               const PolyAdaptiveState &st);

/// reach for alg "poly-adaptive".
PolyReachSet priv_reach_poly_adaptive(const NonlinearSys &sys, const PolyReachProblem &problem,
                                      const alg::PolynomializationAdaptive &alg);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
