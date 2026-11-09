// priv - the private functions of lin_param_sys, as CORA's @linParamSys/private
//
// Not for callers of the class: init_reach, post and the locations of a hybrid automaton use
// them. The series of one step size is computed once and shared by the mappings and the input.
//
// Syntax:   LinParamSeries s = priv_mapping_matrix(sys, r, taylor_terms, intermediate_terms);
// See also: lin_param_sys.h, init_reach.cpp, post.cpp
//
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_dynamics/lin_param_sys/lin_param_sys.h"

#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

/// The series of e^{A r} over the set (CORA's eZ, eI, power.zono, power.int, E): the matrix
/// zonotope part, the interval part, the powers (A r)^i of each (orders 1.., then the orders
/// after them) and the remainder.
struct LinParamSeries {
    MatZonotope zono;
    IntervalMatrix rest;
    std::vector<MatZonotope> zono_powers;
    std::vector<IntervalMatrix> int_powers;
    IntervalMatrix remainder;
};

/// The series of the step r (CORA's priv_mappingMatrix): expmOneParam for a constant set with
/// one generator, else expmMixed (constant) or expmIndMixed (varying) of A r.
LinParamSeries priv_mapping_matrix(const LinParamSys &sys, double r, int taylor_terms,
                                   int intermediate_terms);

/// The power (A r)^i (order i >= 1) of the series as an interval matrix.
IntervalMatrix priv_power(const LinParamSeries &series, int i);

/// The correction matrix F of the curvature between two time points (CORA's priv_tie).
IntervalMatrix priv_tie(const LinParamSeries &series, int taylor_terms);

/// The input of one step (CORA's Rinput, Rtrans and inputCorr).
struct LinParamInput {
    Zonotope input, translation, correction;
};

/// What the inputs in U (centered, m) plus the constant u_trans (m, 1) add over the step r
/// (CORA's priv_inputSolution, with priv_inputTie if B U + c may miss the origin).
LinParamInput priv_input_solution(const LinParamSys &sys, const LinParamSeries &series,
                                  const Zonotope &U, const Tensor &u_trans, double r,
                                  int taylor_terms, int zonotope_order);

/// The mappings of the step size of alg for the inputs U and u_trans.
LinParamMappings priv_mappings(const LinParamSys &sys, const Zonotope &U, const Tensor &u_trans,
                               const alg::LinParamStandard &alg);

/// The first step from R_init (CORA's initReach after the mappings); curvature is F R_init
/// (CORA multiplies F with the first zonotope of a bundle and adds it to every one).
LinParamStep priv_first_step(const Zonotope &R_init, const Zonotope &curvature,
                             LinParamMappings mappings, const alg::LinParamStandard &alg);

/// The step after R (CORA's post without a time-varying input).
LinParamStep priv_next_step(const LinParamStep &R, const alg::LinParamStandard &alg);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
