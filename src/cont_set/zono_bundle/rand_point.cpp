// rand_point - random points of a zonotope bundle, as CORA's zonoBundle.randPoint
//
// Random points of the constrained zonotope of the bundle: "standard" mixes its vertices, "extreme"
// takes support points of random directions. Not a batch.
//
// Syntax:   P = bundle.rand_point(N, rng);   P = bundle.rand_point(N, rng, "extreme");
// Inputs:   N - number of points
//           rng - random numbers
//           type - "standard" or "extreme"
// Outputs:  P - points as columns (n, N)
// See also: con_zonotope, ConZonotope::rand_point
//
// CORA: contSet/@zonoBundle/randPoint_.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/cont_set/private/priv_rand_point.h"
#include "cora/cont_set/con_zonotope/con_zonotope.h"
#include "cora/cont_set/zono_bundle/zono_bundle.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor ZonoBundle::rand_point(int64_t N, Rng &rng, const std::string &type) const {
    detail::Entry entry;
    const detail::Refusal refusal = priv_rand_point_refusal(N, type);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return con_zonotope().rand_point(N, rng, type);
}

Tensor ZonoBundle::rand_point_impl(int64_t N, Rng &rng) const {
    return con_zonotope().rand_point(N, rng);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
