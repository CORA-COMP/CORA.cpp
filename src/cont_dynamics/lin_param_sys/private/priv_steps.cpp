// priv_steps - the mappings and the steps of a linear parametric system, as CORA's initReach/post
//
// The first step maps R by M1 = eZ + mid(eI) and M2 = eI - mid(eI), encloses R and its image
// shifted by Rtrans, adds F R, inputCorr and -Rtrans and the input solution, and reduces. A later
// step maps the set over the previous step by M1 and M2 and adds the input solution; the set at
// the end of a step is mapped the same way, if it is computed.
//
// Syntax:   LinParamMappings m = priv_mappings(sys, U, u_trans, alg);
//           LinParamStep R = priv_first_step(R0, F R0, m, alg);   R = priv_next_step(R, alg);
// Inputs:   sys - the system
//           U, u_trans - the centered input set and the constant input
//           R0 - the initial set; F R0 - the curvature term
//           alg - the algorithm
// Outputs:  m - the mappings of the step size
//           R - the step
// See also: init_reach, post, priv_mapping_matrix, priv_input_solution
//
// CORA: contDynamics/@linParamSys/initReach.m
// CORA: contDynamics/@linParamSys/post.m
// CORA authors: Matthias Althoff
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/lin_param_sys/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The homogeneous map M1 X + M2 X of one step.
Zonotope aux_map(const LinParamMappings &m, const Zonotope &X) {
    return m.mapping * X + m.mapping_rest * X;
}

} // namespace


// ===========================================  MAIN  =========================================== //

LinParamMappings priv_mappings(const LinParamSys &sys, const Zonotope &U, const Tensor &u_trans,
                               const alg::LinParamStandard &alg) {
    const double r = alg.time_step;
    // the series, its correction matrix and the input of the step size
    const LinParamSeries series =
        priv_mapping_matrix(sys, r, alg.taylor_terms, alg.intermediate_terms);
    const IntervalMatrix F = priv_tie(series, alg.taylor_terms);
    LinParamInput in =
        priv_input_solution(sys, series, U, u_trans, r, alg.taylor_terms, alg.zonotope_order);
    // M1 holds the center of the interval part, M2 the rest around it
    const Tensor mid = series.rest.center();
    return {series.zono + mid,   series.rest + mid * (-1.0), F,
            std::move(in.input), std::move(in.translation),  std::move(in.correction)};
}

LinParamStep priv_first_step(const Zonotope &R_init, const Zonotope &curvature,
                             LinParamMappings mappings, const alg::LinParamStandard &alg) {
    const LinParamMappings &m = mappings;
    // the homogeneous solution at the end of the step and over it
    const Zonotope hom_tp = aux_map(m, R_init);
    const Zonotope hom = R_init.enclose(hom_tp + m.translation) + curvature + m.input_correction +
                         (-1.0) * m.translation;
    Zonotope ti = (hom + m.input).reduce(alg.zonotope_order);
    std::optional<Zonotope> tp;
    if (alg.comp_time_point) tp = (hom_tp + m.input).reduce(alg.zonotope_order);
    return {std::move(ti), std::move(tp), std::move(mappings)};
}

LinParamStep priv_next_step(const LinParamStep &R, const alg::LinParamStandard &alg) {
    const LinParamMappings &m = R.mappings;
    // both sets are mapped one step and get the input of the step
    LinParamStep out{(aux_map(m, R.time_interval) + m.input).reduce(alg.zonotope_order),
                     std::nullopt, m};
    if (alg.comp_time_point && R.time_point)
        out.time_point = (aux_map(m, *R.time_point) + m.input).reduce(alg.zonotope_order);
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
