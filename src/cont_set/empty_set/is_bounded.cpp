// is_bounded - whether the empty set is bounded, as CORA's emptySet.isBounded
//
// Syntax:   bool b = E.is_bounded();
// Inputs:   -
// Outputs:  b - true
// See also: is_full_dim, Interval::is_bounded
//
// CORA: contSet/@emptySet/isBounded.m

#include "cora/cont_set/empty_set/empty_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool EmptySet::is_bounded() const {
    detail::Entry entry;
    return true;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
