// linear_map - the image of a zonotope bundle under a matrix, as CORA's zonoBundle.mtimes
//
// Every member is mapped on its own: an enclosure of the image of the intersection (Althoff and
// Krogh, CDC 2011, Prop. 1), exact when the matrix is injective. A scalar factor is ContSet::times.
//
// Syntax:   ZonoBundle R = bundle.linear_map(M);
// Inputs:   M - a matrix (..., k, n)
// Outputs:  R - the bundle of the images of the members
// See also: Zonotope::linear_map, plus
//
// CORA: contSet/@zonoBundle/mtimes.m
// CORA authors: Matthias Althoff, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/zono_bundle/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Every member mapped by M.
ZonoBundle aux_mapped(const std::vector<Zonotope> &members, const Tensor &M) {
    std::vector<Zonotope> mapped;
    mapped.reserve(members.size());
    for (const Zonotope &Z : members) mapped.push_back(Z.linear_map(M));
    return ZonoBundle(std::move(mapped));
}

} // namespace


// ===========================================  MAIN  =========================================== //

ZonoBundle ZonoBundle::linear_map(const Tensor &M) const {
    detail::Entry entry;
    CORA_CHECK(M.ndim() >= 2 && M.shape().back() == dim(), Error::dimension_mismatch, [&] {
        return "linear_map: the matrix needs " + std::to_string(dim()) +
               " columns (the dimension of the set), got the shape " + shape_text(M.shape());
    });
    return aux_mapped(Z_, M);
}

Set ZonoBundle::linear_map_impl(const Tensor &M) const { return aux_mapped(Z_, M); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
