// origin - the ellipsoid that is the origin, as CORA's ellipsoid.origin
//
// The point ellipsoid with Q = 0 and q = 0 in n dimensions.
//
// Syntax:   E = Ellipsoid::origin(n);
// Inputs:   n - the dimension, at least 1
// Outputs:  E - the origin as an ellipsoid
// See also: empty, Capsule::origin
//
// CORA: contSet/@ellipsoid/origin.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

CORA_FROM_NUMBERS Ellipsoid Ellipsoid::origin(int64_t n) {
    detail::Entry entry;
    CORA_CHECK(n >= 1, Error::wrong_value, [&] {
        return "origin: the dimension must be a positive integer, got " + std::to_string(n);
    });
    return {Tensor::zeros({n, n}), Tensor::zeros({n, 1})};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
