// zonotope - a zonotope that contains a capsule, as CORA's capsule.zonotope
//
// The segment is the zonotope of c and g; the ball adds its box, n generators of length r. CORA
// encloses the ball tighter, with 5 n - 1 generators of the equal-area partition (docs/cora-
// differences.md); a radius of 0 adds nothing.
//
// Syntax:   Z = C.zonotope();
// Inputs:   -
// Outputs:  Z - the zonotope of c, g and r times the identity
// See also: ellipsoid, interval
//
// CORA: contSet/@capsule/zonotope.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"
#include "cont_set/capsule/private/priv.h"
#include "cora/cont_set/capsule/capsule.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope Capsule::zonotope() const {
    detail::Entry entry;
    CORA_CHECK(!is_empty(), Error::empty_set,
               [] { return std::string("zonotope: the capsule is empty"); });
    if (priv_all(r_.equal(r_.zeros_like()))) return {c_, g_};
    return {c_, Tensor::concat({g_, r_.multiply(c_.eye_like(dim()))}, -1)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
