// is_full_dim - whether the whole space has a nonempty interior, as CORA's fullspace.isFullDim
//
// Syntax:   bool b = F.is_full_dim();   bool b = F.is_full_dim(tol);
// Inputs:   tol - ignored
// Outputs:  b - true
// See also: Interval::is_full_dim
//
// CORA: contSet/@fullspace/isFullDim.m

#include "cora/cont_set/fullspace/fullspace.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

/// The generic entry has checked tol; the answer does not depend on it.
bool Fullspace::is_full_dim_impl(Tol) const { return true; }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
