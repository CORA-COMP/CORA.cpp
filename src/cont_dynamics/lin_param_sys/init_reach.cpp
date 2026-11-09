// init_reach - the first step of reach of a linear parametric system, as CORA's initReach
//
// The mappings of the step size are computed (priv_mapping_matrix, priv_tie,
// priv_input_solution) and kept in the step for post. The set over the step encloses R_init and
// M1 R_init + M2 R_init + Rtrans, with the curvature F R_init and the input; the set at the end
// is the image plus the input.
//
// Syntax:   LinParamStep R = sys.init_reach(R_init, U, u_trans, alg);
// Inputs:   R_init - the initial set (n)
//           U - the input set (m) without its center (reach removes it)
//           u_trans - the constant input (m, 1)
//           alg - alg::LinParamStandard
// Outputs:  R - the sets of the first step and the mappings
// See also: post, reach
//
// CORA: contDynamics/@linParamSys/initReach.m
// CORA authors: Matthias Althoff
// Authors:       Claude Opus 5.5
// Written:       09-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/lin_param_sys/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

LinParamStep LinParamSys::init_reach(const Zonotope &R_init, const Zonotope &U,
                                     const Tensor &u_trans, const alg::LinParamReach &alg) const {
    detail::Entry entry;
    const auto &a = std::get<alg::LinParamStandard>(alg);
    if (entry.check()) a.validate();
    // the initial set in the state space, the inputs of the columns of B
    CORA_CHECK(R_init.center().shape() == std::vector<int64_t>({dim(), 1}) &&
                   R_init.generators().ndim() == 2,
               Error::dimension_mismatch, [&] {
                   return "init_reach: R_init must be one zonotope of the dimension " +
                          std::to_string(dim()) + ", got the center " +
                          shape_text(R_init.center().shape());
               });
    const int64_t m = B_ ? input_dim() : U.dim();
    CORA_CHECK(U.dim() == m && U.center().ndim() == 2 &&
                   u_trans.shape() == std::vector<int64_t>({m, 1}),
               Error::dimension_mismatch, [&] {
                   return "init_reach: U and u_trans (m, 1) need the " + std::to_string(m) +
                          " columns of B, got " + std::to_string(U.dim()) + " and the shape " +
                          shape_text(u_trans.shape());
               });
    LinParamMappings mappings = priv_mappings(*this, U, u_trans, a);
    const Zonotope curvature = mappings.correction * R_init;
    return priv_first_step(R_init, curvature, std::move(mappings), a);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
