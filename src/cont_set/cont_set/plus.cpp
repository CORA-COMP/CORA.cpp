// plus - the Minkowski sum of two sets of any classes, as CORA's contSet.plus
//
// Syntax:   S = plus(S1, S2);   S = Set(S1) + Set(S2);
// Inputs:   S1, S2 - sets of one dimension
// Outputs:  S - {x + y | x in S1, y in S2}, in the class of the operand of lower precedence
// See also: registry, Zonotope::plus
//
// CORA: contSet/@contSet/plus.m
// CORA authors: Tobias Ladner
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/registry.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set plus(const ContSet &S1, const ContSet &S2) { return detail::dispatch(BinaryOp::plus, S1, S2); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
