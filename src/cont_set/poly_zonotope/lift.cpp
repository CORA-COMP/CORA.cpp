// lift - a polynomial zonotope in a space of higher dimension, with its states at given positions
//
// dims has N positions, each used once: new dimensions would be unbounded, which no polynomial
// zonotope represents. The factors and exponents stay; the center and every generator are
// permuted (CORALean's reindex, exact). CORA's lift_ projects onto dims instead, the inverse
// permutation, as for the zonotope (docs/cora-differences.md).
//
// Syntax:   S = pZ.lift(N, dims);
// Inputs:   N - the dimension of the new space
//           dims - the position (0-based) of each state of pZ in it
// Outputs:  S - the set whose state dims[i] is state i of pZ
// See also: project, zonotope lift
//
// CORA: contSet/@polyZonotope/lift_.m
// CORA: contSet/@contSet/lift.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

PolyZonotope PolyZonotope::lift(int64_t N, const std::vector<int64_t> &dims) const {
    detail::Entry entry;
    const detail::LiftPlan plan = detail::lift_plan(dim(), N, dims);
    // lift_impl comes here after the generic checks, which do not see the permutation: always fail
    if (plan.refusal) CORA_FAIL(plan.refusal->first, plan.refusal->second);
    return {c_.take(plan.source, -1), G_.take(plan.source, -1), GI_.take(plan.source, -1), E_, id_,
            detail::Verbatim{}};
}

Set PolyZonotope::lift_impl(int64_t N, const std::vector<int64_t> &proj) const {
    return lift(N, proj);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
