// con_zonotope - the constrained zonotope of a zonotope bundle, as CORA's zonoBundle.conZonotope
//
// Syntax:   ConZonotope cz = bundle.con_zonotope();
// Inputs:   -
// Outputs:  cz - the first member intersected exactly with the others in order
// (ConZonotope::intersect) See also: ConZonotope::intersect, interval
//
// CORA: contSet/@zonoBundle/conZonotope.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "cora/cont_set/zono_bundle/zono_bundle.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

ConZonotope ZonoBundle::con_zonotope() const {
    detail::Entry entry;
    ConZonotope cz(Z_.front());
    for (std::size_t i = 1; i < Z_.size(); ++i) cz = cz.intersect(Z_[i]);
    return cz;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
