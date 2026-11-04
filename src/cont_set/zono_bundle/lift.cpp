// lift - a zonotope bundle in a space of higher dimension, with its states at given positions
//
// Every member is lifted by Zonotope::lift. CORA's lift_ projects onto dims instead, the inverse
// permutation (docs/cora-differences.md).
//
// Syntax:   ZonoBundle zBl = bundle.lift(N, dims);
// Inputs:   N - the dimension of the new space
//           dims - the position (0-based) of each state; N positions, each used once
// Outputs:  zBl - the bundle whose state dims[i] is state i of bundle
// See also: Zonotope::lift
//
// CORA: contSet/@zonoBundle/lift_.m
//
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zono_bundle/zono_bundle.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

ZonoBundle ZonoBundle::lift(int64_t N, const std::vector<int64_t> &dims) const {
    detail::Entry entry;
    const detail::Refusal r = detail::project_high_dim_refusal(dim(), N, dims, "lift");
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    std::vector<Zonotope> lifted;
    lifted.reserve(Z_.size());
    for (const Zonotope &Z : Z_) lifted.push_back(Z.lift(N, dims));
    return ZonoBundle(std::move(lifted));
}

Set ZonoBundle::lift_impl(int64_t N, const std::vector<int64_t> &proj) const {
    return lift(N, proj);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
