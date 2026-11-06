// quad_map - the quadratic map of a zonotope bundle, as CORA's zonoBundle.quadMap
//
// Every member goes through Zonotope::quad_map: an enclosure of the image of the intersection.
//
// Syntax:   ZonoBundle R = bundle.quad_map(Q);
// Inputs:   Q - the (n, n) matrices of the outputs
// Outputs:  R - the bundle of the images of the members
// See also: Zonotope::quad_map
//
// CORA: contSet/@zonoBundle/quadMap.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/zono_bundle/private/priv.h"
#include "cont_set/zonotope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

ZonoBundle ZonoBundle::quad_map(const std::vector<Tensor> &Q) const {
    detail::Entry entry;
    const std::string problem = priv_quad_map_problem(Q, dim());
    CORA_CHECK(problem.empty(), Error::dimension_mismatch, [&] { return "quad_map: " + problem; });
    std::vector<Zonotope> mapped;
    mapped.reserve(Z_.size());
    for (const Zonotope &Z : Z_) mapped.push_back(Z.quad_map(Q));
    return ZonoBundle(std::move(mapped));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
