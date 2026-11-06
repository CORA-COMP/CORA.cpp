// empty - the empty ellipsoid of n dimensions, as CORA's ellipsoid.empty
//
// Syntax:   Ellipsoid E = Ellipsoid::empty(n);
// Inputs:   n - dimension, n >= 0 (default 0)
// Outputs:  E - the ellipsoid with a shape matrix (0, 0) and a center of the shape (n, 0)
// See also: Ellipsoid::Ellipsoid, EmptySet::empty
//
// CORA: contSet/@ellipsoid/empty.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/ellipsoid/ellipsoid.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

CORA_FROM_NUMBERS Ellipsoid Ellipsoid::empty(int64_t n) {
    detail::Entry entry;
    CORA_CHECK(n >= 0, Error::wrong_value, [&] {
        return "empty: the dimension must be nonnegative, got " + std::to_string(n);
    });
    return {Tensor::zeros({0, 0}), Tensor::zeros({n, 0})};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
