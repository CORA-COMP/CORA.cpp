// reduce - fewer generators for every member of a zonotope bundle, as CORA's zonoBundle.reduce
//
// Every member is reduced on its own, which only enlarges it, so the intersection grows (an
// enclosure). Girard's method is the only one CORA.cpp's zonotope offers; others are
// Error::not_supported.
//
// Syntax:   ZonoBundle R = bundle.reduce(order);   ZonoBundle R = bundle.reduce(order, "girard");
// Inputs:   order - the largest number of generators per dimension, at least 1
//           method - "girard"
// Outputs:  R - a bundle of the reduced members
// See also: Zonotope::reduce
//
// CORA: contSet/@zonoBundle/reduce.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zono_bundle/zono_bundle.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Every member reduced to the order.
ZonoBundle aux_reduced(const std::vector<Zonotope> &members, int order) {
    std::vector<Zonotope> reduced;
    reduced.reserve(members.size());
    for (const Zonotope &Z : members) reduced.push_back(Z.reduce(order));
    return ZonoBundle(std::move(reduced));
}

} // namespace


// ===========================================  MAIN  =========================================== //

ZonoBundle ZonoBundle::reduce(int order, const std::string &method) const {
    detail::Entry entry;
    CORA_CHECK(order >= 1, Error::wrong_value, [&] {
        return "reduce: the zonotope order must be at least 1, got " + std::to_string(order);
    });
    CORA_CHECK(method == "girard", Error::not_supported, [&] {
        return "reduce: the method '" + method + "' is not supported; use \"girard\"";
    });
    return aux_reduced(Z_, order);
}

Set ZonoBundle::reduce_impl(int order) const { return aux_reduced(Z_, order); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
