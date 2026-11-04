// lift - a constrained zonotope in a space of higher dimension, with its states at given positions
//
// Syntax:   cZl = cz.lift(N, dims);
// Inputs:   N - the dimension of the new space;  dims - the position (0-based) of each state of cz
//           in it; N positions, each used once: new dimensions would be unbounded, which no
//           constrained zonotope represents
// Outputs:  cZl - the set whose state dims[i] is state i of cz, exact (a permutation). CORA's
//           lift_ projects onto dims instead, which is the inverse permutation and the same only
//           where dims is its own inverse; CORA.cpp follows the meaning of lift
// See also: project, Zonotope::lift
//
// CORA: contSet/@conZonotope/lift_.m
// CORA: contSet/@contSet/lift.m

#include "cont_set/con_zonotope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

ConZonotope ConZonotope::lift(int64_t N, const std::vector<int64_t> &dims) const {
    detail::Entry entry;
    const detail::LiftPlan plan = detail::lift_plan(dim(), N, dims);
    // lift_impl comes here after the generic checks, which do not see the permutation: always fail
    if (plan.refusal) CORA_FAIL(plan.refusal->first, plan.refusal->second);
    // only the rows of the center and the generators move; the factors and constraints stay
    return {c_.take(plan.source, -2), G_.take(plan.source, -2), A_, b_};
}

Set ConZonotope::lift_impl(int64_t N, const std::vector<int64_t> &proj) const {
    return lift(N, proj);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
