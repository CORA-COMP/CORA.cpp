// post - the step after the last one of reach of a linear parametric system, as CORA's post
//
// The set over the previous step and, if it is computed, the set at its end are mapped by
// M1 + M2 of the step's mappings and get the input solution Rinput; both are reduced. CORA's
// branch for an input that changes every step (uTransVec) is not ported.
//
// Syntax:   LinParamStep Rnext = sys.post(R, alg);
// Inputs:   R - a step of init_reach or post
//           alg - alg::LinParamStandard, the one of init_reach
// Outputs:  Rnext - the sets of the next step, with the same mappings
// See also: init_reach, reach
//
// CORA: contDynamics/@linParamSys/post.m
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

LinParamStep LinParamSys::post(const LinParamStep &R, const alg::LinParamReach &alg) const {
    detail::Entry entry;
    const auto &a = std::get<alg::LinParamStandard>(alg);
    if (entry.check()) a.validate();
    // the step must be one of this system
    CORA_CHECK(R.time_interval.dim() == dim() && R.mappings.mapping.rows() == dim(),
               Error::dimension_mismatch, [&] {
                   return "post: the step must be of the dimension " + std::to_string(dim()) +
                          ", got " + std::to_string(R.time_interval.dim());
               });
    return priv_next_step(R, a);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
