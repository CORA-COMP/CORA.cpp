// cart_prod - the Cartesian product of two sets, as CORA's contSet.cartProd
//
// Syntax:   S = cart_prod(S1, S2);
// Inputs:   S1, S2 - sets of any dimensions
// Outputs:  S - {(x, y) | x in S1, y in S2}, of dimension dim(S1) + dim(S2)
// See also: registry, plus
//
// CORA: contSet/@contSet/cartProd.m
// CORA: contSet/@contSet/cartProd_.m

#include "cora/cont_set/registry.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Set cart_prod(const ContSet &S1, const ContSet &S2) {
    return detail::dispatch(BinaryOp::cart_prod, S1, S2);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
