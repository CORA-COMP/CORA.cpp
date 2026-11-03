// is_full_dim - whether the empty set has a nonempty interior, as CORA's emptySet.isFullDim
//
// Syntax:   bool b = E.is_full_dim();
// Inputs:   -
// Outputs:  b - false
// See also: is_bounded, Interval::is_full_dim
//
// CORA: contSet/@emptySet/isFullDim.m

#include "cora/cont_set/empty_set/empty_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool EmptySet::is_full_dim() const {
    detail::Entry entry;
    return false;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
