// project - the projection of a zonotope bundle onto some dimensions, as CORA's zonoBundle.project
//
// Every member is projected on its own: an enclosure of the projection of the intersection.
//
// Syntax:   Set R = bundle.project(dims);
// Inputs:   dims - the dimensions to keep (0-based), in the order they should appear
// Outputs:  R - the bundle of the projected members
// See also: Zonotope::project
//
// CORA: contSet/@zonoBundle/project.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zono_bundle/zono_bundle.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set ZonoBundle::project_impl(const std::vector<int64_t> &dims) const {
    std::vector<Zonotope> projected;
    projected.reserve(Z_.size());
    for (const Zonotope &Z : Z_) projected.push_back(Z.project(dims).as<Zonotope>());
    return ZonoBundle(std::move(projected));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
