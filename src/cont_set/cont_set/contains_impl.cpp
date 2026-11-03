// contains_impl - whether a set contains another one with no check, as CORA's contSet.contains_
//
// Syntax:   res = detail::contains_impl(S1, S2, tol);
// Inputs:   S1, S2 - sets of one dimension;  tol - as contains (not checked)
// Outputs:  res - what the table has for the classes, as contains; Error::noops without a pair
// See also: contains, registry
//
// CORA: contSet/@contSet/contains_.m

#include "cora/cont_set/registry.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {
namespace detail {


// ===========================================  MAIN  =========================================== //

bool contains_impl(const ContSet &S1, const ContSet &S2, Tol tol) {
    return lookup(BinaryOp::contains, S1, S2, tol);
}

} // namespace detail
} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
