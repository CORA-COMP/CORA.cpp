// origin - the zonotope bundle {0}, as CORA's zonoBundle.origin
//
// Syntax:   ZonoBundle bundle = ZonoBundle::origin(n);
// Inputs:   n - dimension, n >= 1
// Outputs:  bundle - one member without generators, centered at 0
// See also: from, empty
//
// CORA: contSet/@zonoBundle/origin.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zono_bundle/zono_bundle.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

CORA_FROM_NUMBERS ZonoBundle ZonoBundle::origin(int64_t n) {
    detail::Entry entry;
    CORA_CHECK(n >= 1, Error::wrong_value,
               [&] { return "origin: the dimension must be positive, got " + std::to_string(n); });
    return ZonoBundle({Zonotope::origin(n)});
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
