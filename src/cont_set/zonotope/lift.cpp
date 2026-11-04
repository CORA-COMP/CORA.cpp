// lift - a zonotope in a space of higher dimension, with its states at given positions
//
// Syntax:   Zl = Z.lift(N, dims);
// Inputs:   N - the dimension of the new space;  dims - the position (0-based) of each state of Z
//           in it; N positions, each used once: new dimensions would be unbounded, which no
//           zonotope represents
// Outputs:  Zl - the zonotope whose state dims[i] is state i of Z, exact (a permutation).
//           CORA's lift_ projects onto dims instead, which is the inverse permutation and the
//           same only where dims is its own inverse; CORA.cpp follows the meaning of lift
// See also: project
//
// CORA: contSet/@zonotope/lift_.m
// CORA: contSet/@contSet/lift.m

#include "cora/cont_set/zonotope/zonotope.h"


// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::lift(int64_t N, const std::vector<int64_t> &dims) const {
    detail::Entry entry;
    const detail::LiftPlan plan = detail::lift_plan(dim(), N, dims);
    // lift_impl comes here after the generic checks, which do not see the permutation: always fail
    if (plan.refusal) CORA_FAIL(plan.refusal->first, plan.refusal->second);
    return {c_.take(plan.source, -2), G_.take(plan.source, -2)};
}

Set Zonotope::lift_impl(int64_t N, const std::vector<int64_t> &proj) const { return lift(N, proj); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
