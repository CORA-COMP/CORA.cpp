// priv_is_empty_set - whether a set of any class has no point
//
// Syntax:   bool empty = priv_is_empty_set(S);
// Inputs:   S - a set
// Outputs:  empty - true if its center has no columns (n, 0), which is how CORA marks an empty
//           set of any class (CORA's representsa(S, 'emptySet') for the classes that exist)
// See also: contains, isequal

#pragma once

#include "cora/cont_set/cont_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

bool priv_is_empty_set(const ContSet &S);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
