// compact - the empty set is its own compact form, as CORA's contSet.compact does for it
//
// Syntax:   Set R = S.compact();
// Inputs:   -
// Outputs:  R - the set itself; method and tol are ignored, as CORA returns before reading them
// See also: ContSet::compact
//
// CORA: contSet/@contSet/compact.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set EmptySet::compact_impl(const std::optional<std::string> &, Tol) const { return *this; }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
