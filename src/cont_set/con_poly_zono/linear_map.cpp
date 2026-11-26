// linear_map - the image M * cPZ of a constrained polynomial zonotope, as CORA's mtimes
//
// The polynomial part is mapped (PolyZonotope::linear_map: the center and both generator blocks,
// in CORALean's order); the constraint is on the factors and stays.
//
// Syntax:   S = cPZ.linear_map(M);   S = M * cPZ;
// Inputs:   M - (k, n) matrix
// Outputs:  S - the image, k dimensions
// See also: con_poly_zono.h
//
// CORA: contSet/@conPolyZono/mtimes.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_poly_zono/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

ConPolyZono ConPolyZono::linear_map(const Tensor &M) const {
    detail::Entry entry;
    return {poly_.linear_map(M), cons_};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
