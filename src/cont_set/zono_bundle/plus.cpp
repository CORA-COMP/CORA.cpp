// plus - the Minkowski sum of a zonotope bundle with a set or a point, as CORA's zonoBundle.plus
//
// A point or a zonotope is added to every member. A bundle S is added member by member: every pair
// (k1 k2 members, the first bundle's index varying slowest), the enclosure of Althoff and Krogh
// (CDC 2011, Prop. 2). CORA adds the box of S to every member, which is coarser
// (docs/cora-differences.md). A box is its zonotope; the empty box makes the sum empty.
//
// Syntax:   R = bundle.plus(S);   R = bundle.plus(v);
// Inputs:   S - a bundle, a zonotope or a box of the dimension of the bundle
//           v - a column (..., n, 1)
// Outputs:  R - an enclosure of the sum: the intersection of the pairwise sums
// See also: linear_map, Zonotope::plus
//
// CORA: contSet/@zonoBundle/plus.m
// CORA authors: Matthias Althoff
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

/// Every member plus the zonotope Z.
ZonoBundle aux_plus_each(const std::vector<Zonotope> &members, const Zonotope &Z) {
    std::vector<Zonotope> sums;
    sums.reserve(members.size());
    for (const Zonotope &member : members) sums.push_back(member.plus(Z));
    return ZonoBundle(std::move(sums));
}

/// The refusal of the sum with a set of dimension n whose tensors are like `like`, if any.
detail::Refusal aux_refusal(const ZonoBundle &bundle, int64_t n, const Tensor &like) {
    return priv_operand_refusal(bundle, n, like, "plus");
}

} // namespace


// ===========================================  MAIN  =========================================== //

ZonoBundle ZonoBundle::plus(const Zonotope &Z) const {
    detail::Entry entry;
    const detail::Refusal refusal = aux_refusal(*this, Z.dim(), Z.center());
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_plus_each(Z_, Z);
}

ZonoBundle ZonoBundle::plus(const Interval &I) const {
    detail::Entry entry;
    const detail::Refusal refusal = aux_refusal(*this, I.dim(), I.inf());
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    // the empty box has no points to add
    if (I.is_empty()) return empty(dim());
    return aux_plus_each(Z_, I.zonotope());
}

ZonoBundle ZonoBundle::plus(const ZonoBundle &S) const {
    detail::Entry entry;
    const detail::Refusal refusal = aux_refusal(*this, S.dim(), S.zonotopes().front().center());
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    std::vector<Zonotope> sums;
    sums.reserve(Z_.size() * S.zonotopes().size());
    // the pair (i, j) has the index i k2 + j, as the product of the two index sets
    for (const Zonotope &member : Z_)
        for (const Zonotope &other : S.zonotopes()) sums.push_back(member.plus(other));
    return ZonoBundle(std::move(sums));
}

ZonoBundle ZonoBundle::plus(const Tensor &v) const {
    detail::Entry entry;
    CORA_CHECK(v.ndim() >= 2 && v.shape().back() == 1 && v.shape()[v.ndim() - 2] == dim(),
               Error::dimension_mismatch, [&] {
                   return "plus: the point must be a column of " + std::to_string(dim()) +
                          " entries, got the shape " + shape_text(v.shape());
               });
    std::vector<Zonotope> moved;
    moved.reserve(Z_.size());
    for (const Zonotope &Z : Z_) moved.emplace_back(Z.center() + v, Z.generators());
    return ZonoBundle(std::move(moved));
}

Set ZonoBundle::translate_impl(const Tensor &v) const { return plus(v); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
