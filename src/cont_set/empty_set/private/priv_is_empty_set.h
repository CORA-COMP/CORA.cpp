// priv_is_empty_set - whether a set of any class has no point
//
// That is how CORA marks an empty set of any class (CORA's representsa(S, 'emptySet') for the
// classes that exist).
//
// Syntax:   bool empty = priv_is_empty_set(S);
// Inputs:   S - a set
// Outputs:  empty - true if its center has no columns (n, 0)
// See also: contains, isequal
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/cont_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

bool priv_is_empty_set(const ContSet &S);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
