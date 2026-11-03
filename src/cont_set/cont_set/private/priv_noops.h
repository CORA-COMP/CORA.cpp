// priv_noops - the failure of an operation that a set class does not implement
//
// Syntax:   priv_noops("reduce", S);
// Inputs:   operation - the name of the operation;  S - the set of the class without it
// Outputs:  raises Error::noops naming the operation and the class, as CORA's contSet methods do
// See also: cora/cont_set/cont_set.h

#pragma once

#include "cora/cont_set/cont_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

[[noreturn]] void priv_noops(const char *operation, const ContSet &S);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
