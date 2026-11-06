// priv_represents_empty - whether a set is empty, for the classes that have an empty member
//
// False for every other class (a Fullspace and a Zonotope are never empty here).
//
// Syntax:   bool e = priv_represents_empty(S);
// Inputs:   S - a set of any class
// Outputs:  e - true for an EmptySet and an empty interval (bounds without columns)
// See also: plus, is_intersecting
//
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/cont_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

bool priv_represents_empty(const ContSet &S);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
