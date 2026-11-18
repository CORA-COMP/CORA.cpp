// zonotope - a zonotope that contains a zonotope bundle, as CORA's zonoBundle.zonotope
//
// The member of least volume; the volume is the product over the rows of the sum of |G| along the
// row, the volume of the box around the member. The first such member wins a tie.
//
// Syntax:   Zonotope Z = bundle.zonotope();
// Inputs:   -
// Outputs:  Z - a member of the bundle, so a superset of the intersection
// See also: ZonoBundle::interval
//
// CORA: contSet/@zonoBundle/zonotope.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       06-November-2026
// Last update:   18-November-2026
// Reviewed:      ---

#include "cont_set/zono_bundle/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope ZonoBundle::zonotope() const {
    detail::Entry entry;
    // layout: #163 step 2
    const detail::Refusal refusal = priv_batch_refusal(detail::cora_center(Z_.front()), "zonotope");
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    // the index of the least volume is a host decision; the member itself is returned unchanged
    std::size_t best = 0;
    double least = 0.0;
    for (std::size_t i = 0; i < Z_.size(); ++i) {
        // layout: #163 step 2
        const double volume = detail::cora_generators(Z_[i]).abs_sum().prod(-2, true).item();
        if (i == 0 || volume < least) {
            best = i;
            least = volume;
        }
    }
    return Z_[best];
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
