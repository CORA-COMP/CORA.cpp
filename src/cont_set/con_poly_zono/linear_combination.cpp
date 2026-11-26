// linear_combination - the linear combination of two sets, as CORA's conPolyZono.linComb
//
// The polynomial parts combine as PolyZonotope::linear_combination (the factors of both side by
// side, one new factor t last); the constraints join on blocks and do not read t (CORALean's
// linComb: cartProd, then padVars).
//
// Syntax:   S = cPZ.linear_combination(cPZ2);
// Inputs:   cPZ2 - a constrained polynomial zonotope of the same dimension
// Outputs:  S - the points (1 - t)/2 x + (1 + t)/2 y
// See also: con_poly_zono.h
//
// CORA: contSet/@conPolyZono/linComb.m
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

ConPolyZono ConPolyZono::linear_combination(const ConPolyZono &S) const {
    detail::Entry entry;
    CORA_CHECK(S.dim() == dim(), Error::dimension_mismatch,
               [] { return std::string("linear_combination: the sets differ in dimension"); });
    return {poly_.linear_combination(S.poly_), cons_.cart_prod(S.cons_).pad_vars(1)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
