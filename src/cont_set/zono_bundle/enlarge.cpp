// enlarge - a zonotope bundle with every member scaled about its own center
//
// Syntax:   Set R = zB.enlarge(factor);
// Inputs:   factor - a number or a column (..., n, 1): the scaling of every coordinate
// Outputs:  R - the bundle of the enlarged members (each about its own center, so not the
//           enlargement of the intersection about a common center)
// See also: ContSet::enlarge
//
// CORA: contSet/@zonoBundle/enlarge.m

#include "cora/cont_set/zono_bundle/zono_bundle.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set ZonoBundle::enlarge_impl(const Tensor &factor) const {
    std::vector<Zonotope> enlarged;
    for (const Zonotope &Z : Z_) enlarged.push_back(Z.enlarge(factor).as<Zonotope>());
    return ZonoBundle(std::move(enlarged));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
