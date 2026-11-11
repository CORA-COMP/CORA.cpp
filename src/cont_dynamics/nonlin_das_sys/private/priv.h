// priv - the private functions of nonlin_das_sys, as CORA's @nonlinDASys/linReach and helpers
//
// Not for callers of the class: init_reach and post use them. A step linearizes the system,
// eliminates the algebraic variables, solves the affine system, and enlarges the linearization
// error until it holds.
//
// Syntax:   DaeStep s = priv_dae_lin_reach(sys, R, y0, error_x, error_y, U, u_trans, dec, alg);
// See also: nonlin_das_sys.h, reach.cpp, linearize.cpp
//
// CORA.cpp only: the declarations of the private functions of nonlin_das_sys.
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_dynamics/nonlin_das_sys/nonlin_das_sys.h"

#include <optional>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The linearization error of a step (CORA's [Verror, error, error_x, error_y, Rti_y] of
/// priv_linError_mixed_noInt): the set of the error of the states' flow, its bound per state, the
/// bounds of the error of f and of g alone, and the algebraic variables over the step.
struct DaeError {
    Zonotope set;
    Tensor bound, bound_x, bound_y;
    Zonotope algebraic;
};

/// The linearization error over R, the set of the step relative to lin.x_star, with the algebraic
/// error assumed in V_y and the centered input set U0 (CORA's priv_linError_mixed_noInt, or its
/// _comp with a decomposition).
DaeError priv_dae_lin_error(const NonlinDASys &sys, const DaeLinearized &lin, const Zonotope &R,
                            const Zonotope &V_y, const std::optional<Zonotope> &U0,
                            const std::optional<DaeDecomposition> &decomposition,
                            const alg::DaeLinearization &alg);

/// One step from R with the errors of the step before (CORA's linReach of nonlinDASys): the sets
/// over the step and at its end, the algebraic variables, and the errors of this step. y0 is the
/// guess of y at the center of R.
DaeStep priv_dae_lin_reach(const NonlinDASys &sys, const Zonotope &R, const Tensor &y0,
                           const Tensor &error_x, const Tensor &error_y,
                           const std::optional<Zonotope> &U, const Tensor &u_trans,
                           const std::optional<DaeDecomposition> &decomposition,
                           const alg::DaeLinearization &alg);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
