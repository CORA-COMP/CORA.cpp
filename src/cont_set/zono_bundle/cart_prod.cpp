// cart_prod - the Cartesian product of a zonotope bundle with a set, as CORA's zonoBundle.cartProd_
//
// Exact: for a bundle every pair of members, this bundle's the outer index; else member by member.
//
// Syntax:   ZonoBundle R = bundle.cart_prod(S);   ZonoBundle R = ZonoBundle::cart_prod(p, bundle);
// Inputs:   S - a bundle, a zonotope, a nonempty box or a point (n2, 1); the point is first in the
// static form Outputs:  R - the bundle of the products (Zonotope::cart_prod) See also:
// Zonotope::cart_prod, intersect
//
// CORA: contSet/@zonoBundle/cartProd_.m
//
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/zono_bundle/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Why the point is refused, if it is: it is no column.
detail::Refusal aux_point_refusal(const Tensor &p) {
    if (p.ndim() >= 2 && p.shape().back() == 1) return std::nullopt;
    return std::pair{Error::dimension_mismatch,
                     "cart_prod: the point must be a column (n, 1), got the shape " +
                         shape_text(p.shape())};
}

/// The bundle of f(Z) for every member Z.
template <class F>
ZonoBundle aux_map(const ZonoBundle &bundle, F f) {
    std::vector<Zonotope> members;
    for (const Zonotope &Z : bundle.zonotopes()) members.push_back(f(Z));
    return ZonoBundle(std::move(members));
}

} // namespace


// ===========================================  MAIN  =========================================== //

ZonoBundle ZonoBundle::cart_prod(const ZonoBundle &S) const {
    detail::Entry entry;
    const detail::Refusal r =
        priv_operand_refusal(*this, S.dim(), S.zonotopes().front().center(), "cart_prod", false);
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    std::vector<Zonotope> pairs;
    for (const Zonotope &A : Z_)
        for (const Zonotope &B : S.zonotopes()) pairs.push_back(A.cart_prod(B));
    return ZonoBundle(std::move(pairs));
}

ZonoBundle ZonoBundle::cart_prod(const Zonotope &Z) const {
    detail::Entry entry;
    const detail::Refusal r = priv_operand_refusal(*this, Z.dim(), Z.center(), "cart_prod", false);
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    // the zonotope goes behind every member
    return aux_map(*this, [&](const Zonotope &A) { return A.cart_prod(Z); });
}

ZonoBundle ZonoBundle::cart_prod(const Interval &I) const {
    detail::Entry entry;
    const detail::Refusal r = priv_operand_refusal(*this, I.dim(), I.inf(), "cart_prod", false);
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    CORA_CHECK(!I.is_empty(), Error::not_supported,
               [] { return std::string("cart_prod: the empty box has no zonotope"); });
    // so does the box, as its zonotope
    return aux_map(*this, [&](const Zonotope &A) { return A.cart_prod(I); });
}

ZonoBundle ZonoBundle::cart_prod(const Tensor &p) const {
    detail::Entry entry;
    const detail::Refusal r = aux_point_refusal(p);
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    // a point adds rows of zeros to the generators of every member
    return aux_map(*this, [&](const Zonotope &A) { return A.cart_prod(p); });
}

ZonoBundle ZonoBundle::cart_prod(const Tensor &p, const ZonoBundle &bundle) {
    detail::Entry entry;
    const detail::Refusal r = aux_point_refusal(p);
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    // the point goes in front of every member
    return aux_map(bundle, [&](const Zonotope &A) { return Zonotope::cart_prod(p, A); });
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
