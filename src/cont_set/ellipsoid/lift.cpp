// lift - an ellipsoid with its coordinates permuted, as CORA's ellipsoid.lift_
//
// New dimensions would be unbounded, which no ellipsoid represents. CORA's lift_ projects onto
// dims, which is the inverse permutation and the same only where dims is its own inverse; CORA.cpp
// follows the meaning of lift.
//
// Syntax:   El = E.lift(N, dims);
// Inputs:   N - the dimension of the new space, N = dim()
//           dims - the position of each coordinate, a permutation
// Outputs:  El - the permuted ellipsoid
// See also: project, Capsule::lift
//
// CORA: contSet/@ellipsoid/lift_.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Ellipsoid Ellipsoid::lift(int64_t N, const std::vector<int64_t> &dims) const {
    detail::Entry entry;
    const detail::LiftPlan plan = detail::lift_plan(dim(), N, dims);
    // lift_impl comes here after the generic checks, which do not see the permutation: always fail
    if (plan.refusal) CORA_FAIL(plan.refusal->first, plan.refusal->second);
    if (priv_is_empty(*this)) return *this;
    return {Q_.take(plan.source, -2).take(plan.source, -1), q_.take(plan.source, -2), tol_};
}

Set Ellipsoid::lift_impl(int64_t N, const std::vector<int64_t> &proj) const {
    return lift(N, proj);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
