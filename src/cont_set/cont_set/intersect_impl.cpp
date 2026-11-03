// intersect_impl - the intersection of two sets with no check, as CORA's contSet.and_
//
// Syntax:   S = detail::intersect_impl(S1, S2);
// Inputs:   S1, S2 - sets of one dimension (not checked)
// Outputs:  S - what the table has for the classes, as intersect; Error::noops without a pair
// See also: intersect, registry
//
// CORA: contSet/@contSet/and_.m

#include "cora/cont_set/registry.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {
namespace detail {


// ===========================================  MAIN  =========================================== //

Set intersect_impl(const ContSet &S1, const ContSet &S2) {
    return lookup(BinaryOp::intersect, S1, S2);
}

} // namespace detail
} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
