// lift - a capsule with its coordinates permuted, as CORA's capsule.lift_
//
// Syntax:   Cl = C.lift(N, dims);
// Inputs:   N - the dimension of the new space, N = dim(): new dimensions would be unbounded,
//           which no capsule represents;  dims - the position (0-based) of each coordinate of C,
//           a permutation of 0..N-1
// Outputs:  Cl - the capsule whose coordinate dims[i] is coordinate i of C: center and
//           generator permuted, the radius kept. CORA's lift_ projects onto dims, which is the
//           inverse permutation and the same only where dims is its own inverse; CORA.cpp
//           follows the meaning of lift
// See also: Zonotope::lift
//
// CORA: contSet/@capsule/lift_.m
// CORA: contSet/@contSet/lift.m

#include "cora/cont_set/capsule/capsule.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Capsule Capsule::lift(int64_t N, const std::vector<int64_t> &dims) const {
    detail::Entry entry;
    const detail::LiftPlan plan = detail::lift_plan(dim(), N, dims);
    // lift_impl comes here after the generic checks, which do not see the permutation: always fail
    if (plan.refusal) CORA_FAIL(plan.refusal->first, plan.refusal->second);
    return {Unchecked{}, c_.take(plan.source, -2), g_.take(plan.source, -2), r_};
}

Set Capsule::lift_impl(int64_t N, const std::vector<int64_t> &proj) const { return lift(N, proj); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
