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

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
