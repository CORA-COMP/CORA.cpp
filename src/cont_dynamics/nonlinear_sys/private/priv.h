// priv - the private functions of nonlinear_sys, as CORA's linReach and its private helpers
//
// Not for callers of the class: init_reach and post use them. A step linearizes the system,
// solves the affine system, and enlarges the linearization error until it holds.
//
// Syntax:   NonlinearStep s = priv_lin_reach(sys, R, error, U, alg);
// See also: nonlinear_sys.h, reach.cpp, linearize.cpp
//
// CORA.cpp only: the declarations of the private functions of nonlinear_sys.
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cont_dynamics/linear_sys/private/priv_steps.h"
#include "cora/cont_dynamics/nonlinear_sys/nonlinear_sys.h"

#include <optional>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// The error of a step -----------------------------------------------------------------------------

/// The linearization error of a step (CORA's trueError and VerrorDyn): the bound per state (n, 1)
/// and the set of the error.
struct AbstractionError {
    Tensor bound;
    Zonotope set;
};

// The affine step ---------------------------------------------------------------------------------

/// One step of the affine system x' = A x + u, with what the solutions of the error of the step
/// reuse: the powers of A and the block structure (the step of nonlinear and algebraic systems).
struct AffineStep {
    TaylorLinSys taylor;
    Blocks blocks;
    Zonotope time_interval, time_point;
};

/// The step of sys from R_delta (relative to the expansion point) with the input set U and the
/// constant input u_const, in n = dim states (CORA's oneStep of linReach).
AffineStep priv_affine_step(const LinearSys &sys, const Zonotope &R_delta, const Zonotope &U,
                            const Tensor &u_const, double time_step, int taylor_terms);

/// The input set U without its center (CORA's params.U after the split into U and uTrans); without
/// a set, the input 0; none for a system without inputs (m = 0). `like` gives the backend.
std::optional<Zonotope> priv_centered_inputs(const Tensor &like, int64_t m,
                                             const std::optional<Zonotope> &U);

/// The failure of a linearization error beyond 1e100: the reachable set explodes.
void priv_check_exploded(const Tensor &error);

// The errors and the steps ------------------------------------------------------------------------

/// One step from R whose linearization error was `error` in the step before (CORA's linReach
/// with alg "lin"): the sets over the step, at its end, and the error of this step.
NonlinearStep priv_lin_reach(const NonlinearSys &sys, const Zonotope &R, const Tensor &error,
                             const std::optional<Zonotope> &U, const alg::Linearization &alg);

/// The linearization error over R, a set relative to lin.x_star, with the centered input set U0
/// (CORA's priv_abstrerr_lin, tensorOrder 2 or 3).
AbstractionError priv_abstrerr_lin(const NonlinearSys &sys, const Zonotope &R,
                                   const Linearized &lin, const std::optional<Zonotope> &U0,
                                   const alg::Linearization &alg);

/// The error of a step with alg "poly" (CORA's trueError, VerrorDyn): the bound per state (n, 1)
/// and the zonotope of the dynamic part.
struct PolyAbstractionError {
    Tensor bound;
    Zonotope set;
};

/// What priv_poly_reach computes once per step for the static error (CORA's priv_precompStatError):
/// the Hessians H at the expansion point, the reduced set Zdelta of the states with the inputs 0,
/// and the static error as a polynomial zonotope.
struct PolyStaticError {
    std::vector<Tensor> H;
    Zonotope Zdelta;
    PolyZonotope error;
};

// The steps ---------------------------------------------------------------------------------------

/// The loop of CORA's linReach that every algorithm shares: the error is guessed as 1.1 times
/// `error`, error_over(R_guess) computes the error over the set that the guess allows (R_guess is
/// the solution of the guess), and while it exceeds the guess somewhere it becomes the guess.
AbstractionError
priv_error_loop(const Linearized &lin, TaylorLinSys &taylor, const Tensor &error, double dt,
                int eta,
                const std::function<AbstractionError(const Zonotope &R_guess)> &error_over);

/// What the steps "lin" and "poly" from a polynomial zonotope share (CORA's linReach before the
/// error): the affine system at the center, R - x*, and its solution at the end of the step
/// (R_tp) and over it (R_ti), without the error.
struct PolyStepPieces {
    PolyStepPieces(const NonlinearSys &sys, const PolyZonotope &R, const std::optional<Zonotope> &U,
                   double dt, int eta);

    Linearized lin;
    PolyZonotope R_delta;
    TaylorLinSys taylor;
    Blocks blocks;
    Zonotope PU;
    BlockConstant Pu;
    PolyZonotope R_tp, R_ti;
    std::optional<Zonotope> U0; ///< the centered input set (CORA's params.U), none without inputs
};

/// One step of "lin" from a polynomial zonotope (CORA's linReach with alg 'lin' and a
/// polyZonotope): the dependency stays through the linear part, the error is a zonotope.
PolyNonlinearStep priv_lin_reach_poly(const NonlinearSys &sys, const PolyZonotope &R,
                                      const Tensor &error, const std::optional<Zonotope> &U,
                                      const alg::Linearization &alg);

/// The enclosure of the segments between the points of A and B, which have the factors and the
/// exponents of A (CORALean's encloseShared): the factors of A and one new one.
PolyZonotope priv_poly_enclose_shared(const PolyZonotope &A, const PolyZonotope &B);

/// CORA's constructor as CORALean's construct in the order of the first occurrences: zero
/// generators dropped, generators of equal exponents added.
PolyZonotope priv_poly_construct(const PolyZonotope &Z);

/// CORA's zonotope(pZ) as CORALean's zonotopeFold: even generators halved, constant ones and the
/// other halves added into the center.
Zonotope priv_poly_zonotope_fold(const PolyZonotope &Z);

/// CORA's reduce(pZ, 'girard', order) as CORALean's reduceCora: past order * n generators the
/// shortest are boxed into the independent part; the factors no generator reads are dropped.
PolyZonotope priv_poly_reduce_cora(const PolyZonotope &Z, int order);

/// One step from the polynomial zonotope R (CORA's linReach with alg "poly").
PolyNonlinearStep priv_poly_reach(const NonlinearSys &sys, const PolyZonotope &R,
                                  const Tensor &error, const std::optional<Zonotope> &U,
                                  const alg::Polynomialization &alg);

/// The static part of the error over Rdelta, a set relative to lin.x_star (CORA's
/// priv_precompStatError).
PolyStaticError priv_precomp_stat_error(const NonlinearSys &sys, const PolyZonotope &Rdelta,
                                        const Linearized &lin, bool has_inputs,
                                        const alg::Polynomialization &alg);

/// The error over Rall, with Rdiff the set of the differences of the states over the step (CORA's
/// priv_abstrerr_poly, tensorOrder 3).
PolyAbstractionError priv_abstrerr_poly(const NonlinearSys &sys, const PolyZonotope &Rall,
                                        const PolyZonotope &Rdiff, const Linearized &lin,
                                        const std::optional<Zonotope> &U0,
                                        const PolyStaticError &stat,
                                        const alg::Polynomialization &alg);

/// The box 1/6 sum_j dz_j (dz' T_ij dz) of the third-order remainder over the box IHx of the
/// states (relative to lin.x_star) and IHu of the inputs, per component; none if the third
/// derivatives are identically 0.
std::optional<Interval> priv_third_order_box(const NonlinearSys &sys, const Interval &IHx,
                                             const std::optional<Interval> &IHu,
                                             const Linearized &lin);

/// The box joined with the origin, which is the linearization point in coordinates relative to it:
/// the region of a Lagrange remainder must contain the point it expands at (CORALean's repair).
Interval priv_with_origin(const Interval &I);

/// MATLAB's max(true_error ./ applied): NaN entries (0/0) are skipped, NaN if all are.
double priv_ratio_max(const Tensor &true_error, const Tensor &applied);

/// Whether some entry is above CORA's bound of an exploding error, 1e100.
bool priv_exploded(const Tensor &error);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
