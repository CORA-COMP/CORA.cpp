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

// The interval arithmetic of the remainders (priv_remainder.cpp) --------------------------------

/// The entries of a column (n, 1) as n tensors of shape (1, 1), and back.
std::vector<Tensor> priv_entries(const Tensor &column);
Tensor priv_column(const std::vector<Tensor> &entries);

/// max(|inf|, |sup|) of a box: CORA's max(abs(infimum(I)), abs(supremum(I))).
Tensor priv_abs_max(const Tensor &lo, const Tensor &hi);

/// The bounds of z - z* as ranges (1, 1): the box of R, then that of U0.
std::vector<Range> priv_dz(const Interval &IHx, const std::optional<Interval> &IHu);

/// MATLAB's max(numerator ./ denominator): NaN entries (0/0) are skipped, NaN if all are.
double priv_ratio_max(const Tensor &numerator, const Tensor &denominator);

/// tensorOrder 2: 1/2 dz' max|H_i| dz per component over the nonzero entries of the Hessian;
/// dz holds the largest |z - z*| per coordinate as (1, 1) tensors.
Tensor priv_second_order(const std::vector<std::optional<Range>> &H, const std::vector<Tensor> &dz,
                         int64_t n);

/// tensorOrder 3: the box 1/6 sum_j dz_j (dz' T_ij dz) per component, over the j with an entry of
/// T_ij that is not identically 0 (CORA's ind{i}); none if there is no such entry at all. With
/// by_position, dz_j is the dz of the place of j in ind{i} (as CORA's adaptive error does).
std::optional<Interval> priv_third_order(const std::vector<std::optional<Range>> &T,
                                         const std::vector<Range> &dz, int64_t n, bool by_position);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
