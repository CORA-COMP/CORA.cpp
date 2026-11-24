// priv - the private functions of nonlinear_sys_dt, as CORA's linReach and its private helpers
//
// Not for callers of the class: post and reach use them.
//
// Syntax:   DiscreteStep s = priv_lin_reach(sys, R, U, alg);
// See also: nonlinear_sys_dt.h, reach.cpp
//
// CORA.cpp only: the declarations of the private functions of nonlinear_sys_dt.
// Authors:       Claude Sonnet 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cont_dynamics/nonlinear_sys/private/priv.h"
#include "cora/cont_dynamics/nonlinear_sys_dt/nonlinear_sys_dt.h"

#include <optional>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The set of the inputs of a system: U, or the zero input if none is given; none without inputs.
std::optional<Zonotope> priv_input_set(const NonlinearSysDT &sys, const Tensor &like,
                                       const std::optional<Zonotope> &U);

/// One step from R (CORA's linReach): linearized at the center of R and of U, plus the error of
/// the tensor order, reduced to zonotope_order; the set and the error.
DiscreteStep priv_lin_reach(const NonlinearSysDT &sys, const Zonotope &R,
                            const std::optional<Zonotope> &U,
                            const alg::DiscreteLinearization &alg);

/// The linearization error of the step map over Rdelta (relative to the center of R) and the
/// centered input set U0 (CORA's priv_linError_mixed_noInt, tensor order 2, and
/// priv_linError_thirdOrder, tensor order 3), reduced to zonotope_order; p_x, p_u the point.
Zonotope priv_lin_error(const NonlinearSys &f, const Zonotope &Rdelta,
                        const std::optional<Zonotope> &U0, const Tensor &p_x, const Tensor &p_u,
                        const alg::DiscreteLinearization &alg);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
