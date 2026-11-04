// intersect - the intersection of a zonotope bundle with a set, as CORA's zonoBundle.and_
//
// Exact: the result lists the members of both, which is what a bundle denotes.
//
// Syntax:   ZonoBundle R = bundle.intersect(S);
// Inputs:   S - a bundle, a zonotope or a nonempty box of the same dimension and batch shape
// Outputs:  R - the bundle of the members of bundle and of S (a box as its zonotope)
// See also: con_zonotope, Zonotope::intersect
//
// CORA: contSet/@zonoBundle/and_.m
//
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/zono_bundle/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The members of bundle followed by the zonotopes `more`.
ZonoBundle aux_appended(const ZonoBundle &bundle, const std::vector<Zonotope> &more) {
    std::vector<Zonotope> members = bundle.zonotopes();
    members.insert(members.end(), more.begin(), more.end());
    return ZonoBundle(std::move(members));
}

} // namespace


// ===========================================  MAIN  =========================================== //

ZonoBundle ZonoBundle::intersect(const ZonoBundle &S) const {
    detail::Entry entry;
    const detail::Refusal r =
        priv_operand_refusal(*this, S.dim(), S.zonotopes().front().center(), "intersect");
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    return aux_appended(*this, S.zonotopes());
}

ZonoBundle ZonoBundle::intersect(const Zonotope &Z) const {
    detail::Entry entry;
    const detail::Refusal r = priv_operand_refusal(*this, Z.dim(), Z.center(), "intersect");
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    return aux_appended(*this, {Z});
}

ZonoBundle ZonoBundle::intersect(const Interval &I) const {
    detail::Entry entry;
    const detail::Refusal r = priv_operand_refusal(*this, I.dim(), I.inf(), "intersect");
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    CORA_CHECK(!I.is_empty(), Error::not_supported,
               [] { return std::string("intersect: the empty box has no zonotope"); });
    return aux_appended(*this, {I.zonotope()});
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
