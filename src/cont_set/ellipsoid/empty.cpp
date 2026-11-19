// empty - the empty ellipsoid of n dimensions, as CORA's ellipsoid.empty
//
// Syntax:   Ellipsoid E = Ellipsoid::empty(n);
// Inputs:   n - dimension, n >= 0 (default 0)
// Outputs:  E - the ellipsoid of the center 0 (n) and Q = 0 (n, n), flagged empty in empty_mask()
// See also: Ellipsoid::Ellipsoid, EmptySet::empty
//
// CORA: contSet/@ellipsoid/empty.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   19-November-2026
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
    Ellipsoid E(Tensor::zeros({n}), Tensor::zeros({n, n}));
    E.empty_ = EmptyMask::all_members();
    return E;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
