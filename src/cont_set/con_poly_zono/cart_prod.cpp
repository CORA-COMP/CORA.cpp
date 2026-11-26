// cart_prod - the Cartesian product of constrained polynomial zonotopes, as CORA's cartProd
//
// Each set keeps its factors and its constraint, on blocks: the centers stacked, the generator rows
// block-diagonal with +0 elsewhere (CORALean's genBlockDiag), the exponents and the constraints
// side by side (CORALean's cartProd); the identifiers are 1..p.
//
// Syntax:   S = cPZ.cart_prod(cPZ2);
// Inputs:   cPZ2 - a constrained polynomial zonotope of any dimension
// Outputs:  S - the product
// See also: con_poly_zono.h
//
// CORA: contSet/@conPolyZono/cartProd_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_poly_zono/private/priv.h"
#include "cora/global/error.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

ConPolyZono ConPolyZono::cart_prod(const ConPolyZono &S) const {
    detail::Entry entry;
    CORA_CHECK(dim() > 0 && S.dim() > 0, Error::wrong_value,
               [] { return std::string("cart_prod: a set of no dimension has no product"); });
    const int64_t p1 = int64_t(ids().size()), p2 = int64_t(S.ids().size());
    return priv_set(Tensor::concat({c(), S.c()}, 0), priv_block_rows(generators(), S.generators()),
                    priv_block_rows(independent_generators(), S.independent_generators()),
                    priv_block_exponents(exponents(), p1, S.exponents(), p2),
                    cons_.cart_prod(S.cons_));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
