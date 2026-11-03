// is_bounded - whether a constrained zonotope is bounded
//
// Syntax:   res = cz.is_bounded();
// Inputs:   -
// Outputs:  res - always true: the generators are applied to factors of the unit box
// See also: is_full_dim, Zonotope::is_bounded
//
// CORA: contSet/@conZonotope/isBounded.m

#include "cora/cont_set/con_zonotope/con_zonotope.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool ConZonotope::is_bounded() const {
    detail::Entry entry;
    return true;
}

bool ConZonotope::is_bounded_impl() const { return is_bounded(); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
