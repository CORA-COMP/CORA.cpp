// priv_represents_empty - whether a set is empty, for the classes that have an empty member
//
// Syntax:   bool e = priv_represents_empty(S);
// Inputs:   S - a set of any class
// Outputs:  e - true for an empty interval (bounds without columns), false for every other class
//           (a Fullspace and a Zonotope are never empty here)
// See also: plus, is_intersecting

#pragma once

#include "cora/cont_set/cont_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

bool priv_represents_empty(const ContSet &S);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
