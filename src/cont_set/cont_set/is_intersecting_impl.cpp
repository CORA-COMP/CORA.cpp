// is_intersecting_impl - whether two sets intersect, unchecked, as CORA's isIntersecting_
//
// Syntax:   res = detail::is_intersecting_impl(S1, S2, tol);
// Inputs:   S1, S2 - sets of one dimension;  tol - as is_intersecting (not checked)
// Outputs:  res - what the table has for the classes, as is_intersecting; Error::noops without
//           a pair
// See also: is_intersecting, registry
//
// CORA: contSet/@contSet/isIntersecting_.m

#include "cora/cont_set/registry.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {
namespace detail {


// ===========================================  MAIN  =========================================== //

bool is_intersecting_impl(const ContSet &S1, const ContSet &S2, Tol tol) {
    return lookup(BinaryOp::is_intersecting, S1, S2, tol);
}

} // namespace detail
} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
