// enlarge - a zonotope bundle with every member scaled about its own center
//
// Not the enlargement of the intersection about a common center.
//
// Syntax:   Set R = bundle.enlarge(factor);
// Inputs:   factor - a number or a column (..., n, 1): the scaling of every coordinate
// Outputs:  R - the bundle of the enlarged members
// See also: ContSet::enlarge
//
// CORA: contSet/@zonoBundle/enlarge.m
//
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zono_bundle/zono_bundle.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set ZonoBundle::enlarge_impl(const Tensor &factor) const {
    std::vector<Zonotope> enlarged;
    enlarged.reserve(Z_.size());
    for (const Zonotope &Z : Z_) enlarged.push_back(Z.enlarge(factor).as<Zonotope>());
    return ZonoBundle(std::move(enlarged));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
