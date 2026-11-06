// is_bounded - whether an ellipsoid is bounded
//
// Syntax:   res = E.is_bounded();
// Inputs:   -
// Outputs:  res - always true: the shape matrix is finite
// See also: is_full_dim, Zonotope::is_bounded
//
// CORA: contSet/@ellipsoid/isBounded.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/ellipsoid/ellipsoid.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool Ellipsoid::is_bounded() const {
    detail::Entry entry;
    return true;
}

bool Ellipsoid::is_bounded_impl() const { return is_bounded(); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
