// from - a zonotope or a box as a zonotope bundle, as CORA's zonoBundle(Z) conversions
//
// A bundle of one member; a member of CORA.cpp is a zonotope, so a box goes through its zonotope.
//
// Syntax:   ZonoBundle bundle = ZonoBundle::from(Z);   ZonoBundle bundle = ZonoBundle::from(I);
// Inputs:   Z - a zonotope
//           I - a box (nonempty)
// Outputs:  bundle - the same set
// See also: ZonoBundle::ZonoBundle, Interval::zonotope
//
// CORA: contSet/@zonotope/zonoBundle.m
// CORA: contSet/@interval/zonoBundle.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zono_bundle/zono_bundle.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

ZonoBundle ZonoBundle::from(const Zonotope &Z) {
    detail::Entry entry;
    return ZonoBundle({Z});
}

ZonoBundle ZonoBundle::from(const Interval &I) {
    detail::Entry entry;
    CORA_CHECK(!I.is_empty(), Error::not_supported,
               [] { return std::string("from: the empty box has no zonotope"); });
    return ZonoBundle({I.zonotope()});
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
