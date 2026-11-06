// is_bounded - whether a zonotope bundle is bounded
//
// Syntax:   res = bundle.is_bounded();
// Inputs:   -
// Outputs:  res - always true: the members are zonotopes
// See also: is_full_dim, Zonotope::is_bounded
//
// CORA: contSet/@zonoBundle/isBounded.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zono_bundle/zono_bundle.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool ZonoBundle::is_bounded() const {
    detail::Entry entry;
    return true;
}

bool ZonoBundle::is_bounded_impl() const { return is_bounded(); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
