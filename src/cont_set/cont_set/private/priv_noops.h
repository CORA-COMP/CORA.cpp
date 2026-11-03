// priv_noops - the failure of an operation that a set class does not implement
//
// Syntax:   priv_noops("reduce", S);   priv_noops("enclose", S1, &S2);
// Inputs:   operation - the name of the operation;  S - the set of the class without it;
//           other - the second operand of an operation on a pair of sets
// Outputs:  raises Error::noops naming the operation and the class (the classes), as CORA does
// See also: cora/cont_set/cont_set.h

#pragma once

#include "cora/cont_set/cont_set.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

[[noreturn]] void priv_noops(const char *operation, const ContSet &S,
                             const ContSet *other = nullptr);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
