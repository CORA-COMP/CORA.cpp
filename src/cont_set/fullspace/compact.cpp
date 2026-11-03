// compact - the whole space is its own compact form, as CORA's contSet.compact does for it
//
// Syntax:   Set R = S.compact();
// Inputs:   -
// Outputs:  R - the set itself; method and tol are ignored, as CORA returns before reading them
// See also: ContSet::compact
//
// CORA: contSet/@contSet/compact.m

#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/cont_set/set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set Fullspace::compact_impl(const std::optional<std::string> &, Tol) const { return *this; }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
