// is_full_dim - whether the empty set has a nonempty interior, as CORA's emptySet.isFullDim
//
// Syntax:   bool b = E.is_full_dim();   bool b = E.is_full_dim(tol);
// Inputs:   tol - ignored
// Outputs:  b - false
// See also: is_bounded, Interval::is_full_dim
//
// CORA: contSet/@emptySet/isFullDim.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

/// The generic entry has checked tol; the answer does not depend on it.
bool EmptySet::is_full_dim_impl(Tol) const { return false; }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
