// plus - the Minkowski sum of two sets of any classes, as CORA's contSet.plus
//
// Syntax:   S = plus(S1, S2);   S = Set(S1) + Set(S2);
// Inputs:   S1, S2 - sets of one dimension
// Outputs:  S - {x + y | x in S1, y in S2}, in the class of the operand of lower precedence
// See also: registry, Zonotope::plus
//
// CORA: contSet/@contSet/plus.m

#include "cora/cont_set/registry.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set plus(const ContSet &S1, const ContSet &S2) {
    detail::Entry entry;
    CORA_CHECK(S1.dim() == S2.dim(), Error::dimension_mismatch, [&] {
        return "plus: the sets differ in dimension, " + S1.class_name() + " of " +
               std::to_string(S1.dim()) + " and " + S2.class_name() + " of " +
               std::to_string(S2.dim());
    });
    return detail::dispatch(BinaryOp::plus, S1, S2);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
