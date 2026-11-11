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

#include "cora/cont_dynamics/nonlinear_sys/nonlinear_sys.h"

#include <optional>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The linearization error of a step (CORA's trueError and VerrorDyn): the bound per state (n, 1)
/// and the set of the error.
struct AbstractionError {
    Tensor bound;
    Zonotope set;
};

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

/// MATLAB's max(true_error ./ applied): NaN entries (0/0) are skipped, NaN if all are.
double priv_ratio_max(const Tensor &true_error, const Tensor &applied);

/// Whether some entry is above CORA's bound of an exploding error, 1e100.
bool priv_exploded(const Tensor &error);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
