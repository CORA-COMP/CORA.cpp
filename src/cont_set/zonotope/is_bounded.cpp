// is_bounded - whether a zonotope is bounded
//
// Syntax:   res = Z.is_bounded();
// Inputs:   -
// Outputs:  res - always true: a zonotope is the image of a cube
// See also: is_full_dim
//
// CORA: contSet/@zonotope/isBounded.m
// CORA authors: Mark Wetzlinger
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/zonotope/zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool Zonotope::is_bounded() const {
    detail::Entry entry;
    return true;
}

bool Zonotope::is_bounded_impl() const { return is_bounded(); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
