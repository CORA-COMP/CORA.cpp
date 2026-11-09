// lin_param_sys - the linear system x' = A x + B u + c, A in a matrix set, as CORA's linParamSys
//
// A is a matrix zonotope (n, n): every matrix in it is a possible system matrix, constant over
// time ("constParam", ParamType::constant) or changing arbitrarily ("varParam",
// ParamType::varying). B (n, m) and c (n, 1) are optional matrices of numbers. reach runs CORA's
// initReach and post step by step; the mappings of one step size are computed once.
//
// Syntax:     LinParamSys sys(A, B);   LinParamSys sys(A, B, c, ParamType::varying);
//             ReachSet R = sys.reach({.initial_set = X0, .t_final = 2}, alg::LinParamStandard{});
// Operations: reach, init_reach, post (one file each); the mappings of a step are in private/;
//             the constructor checks its input (lin_param_sys.cpp)
// See also:   cora/cont_dynamics/linear_sys/linear_sys.h, mat_zonotope.h
//
// CORA: contDynamics/@linParamSys/linParamSys.m
// CORA authors: Matthias Althoff
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_dynamics/algorithms.h"
#include "cora/cont_dynamics/cont_dynamics.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/matrix_set/interval_matrix/interval_matrix.h"
#include "cora/matrix_set/mat_zonotope/mat_zonotope.h"

#include <optional>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// Whether the matrix of a LinParamSys is one of the set for all time (CORA's 'constParam') or
/// may change at every instant ('varParam').
enum class ParamType { constant, varying };

/// What one step size makes of the system and its input (CORA keeps these in the system object:
/// mappingMatrixSet.zono and .int, F, Rinput, Rtrans and inputCorr).
struct LinParamMappings {
    MatZonotope mapping; ///< the series to its matrix-zonotope order, with the rest's center
    IntervalMatrix mapping_rest; ///< the rest of the series around its center
    IntervalMatrix correction;   ///< F: the curvature between two time points
    Zonotope input;              ///< Rinput: the input solution of one step
    Zonotope translation;        ///< Rtrans: the part of the constant input
    Zonotope input_correction;   ///< inputCorr: the curvature of the constant input
};

/// One step of reach (CORA's Rfirst/Rnext): the set over the step, the set at its end if it is
/// computed, and the mappings of the step size.
struct LinParamStep {
    Zonotope time_interval;
    std::optional<Zonotope> time_point;
    LinParamMappings mappings;
};

class LinParamSys {
  public:
    /// x' = A x + B u + c for the matrices A (n, n) of the set; B (n, m) and c (n, 1) optional.
    explicit LinParamSys(MatZonotope A, std::optional<Tensor> B = std::nullopt,
                         std::optional<Tensor> c = std::nullopt,
                         ParamType type = ParamType::constant);

    const MatZonotope &A() const noexcept { return A_; }
    const std::optional<Tensor> &B() const noexcept { return B_; }
    const Tensor &c() const noexcept { return c_; }
    ParamType type() const noexcept { return type_; }

    /// The number of states n, and of inputs (the columns of B, 0 without B).
    int64_t dim() const { return A_.rows(); }
    int64_t input_dim() const { return B_ ? B_->shape()[1] : 0; }

    /// The reachable sets from problem.initial_set over the steps of t_final / time_step, for
    /// every u(t) in problem.input_set (needs B) if there is one. time_point is empty if the
    /// algorithm does not compute it.
    ReachSet reach(const ReachProblem &problem, const alg::LinParamReach &alg) const;

    /// The first step from R_init for the inputs in U plus the constant u_trans (m, 1); U should
    /// hold the origin, as reach makes it (CORA's initReach).
    LinParamStep init_reach(const Zonotope &R_init, const Zonotope &U, const Tensor &u_trans,
                            const alg::LinParamReach &alg) const;

    /// The step after R, with its mappings (CORA's post).
    LinParamStep post(const LinParamStep &R, const alg::LinParamReach &alg) const;

  private:
    MatZonotope A_;
    std::optional<Tensor> B_;
    Tensor c_;
    ParamType type_;
};

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
