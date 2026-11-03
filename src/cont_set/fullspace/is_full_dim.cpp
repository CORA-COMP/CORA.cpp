// is_full_dim - whether the whole space has a nonempty interior, as CORA's fullspace.isFullDim
//
// Syntax:   bool b = F.is_full_dim();
// Inputs:   -
// Outputs:  b - true
// See also: Interval::is_full_dim
//
// CORA: contSet/@fullspace/isFullDim.m

#include "cora/cont_set/fullspace/fullspace.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool Fullspace::is_full_dim() const {
    detail::Entry entry;
    return true;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
