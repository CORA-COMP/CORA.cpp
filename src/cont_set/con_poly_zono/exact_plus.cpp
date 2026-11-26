// exact_plus - the sum over the same factors, as CORA's exactPlus
//
// Both sets have the same identifiers and the same constraint: the centers add and the generators
// join on the shared factors (CORALean's exactPlus, the constraint of the first set). CORA aligns
// the identifiers first and refuses other constraints; here both are refused if they differ.
//
// Syntax:   S = cPZ.exact_plus(cPZ2);
// Inputs:   cPZ2 - a set of the same dimension, identifiers and constraint
// Outputs:  S - the sum
// See also: con_poly_zono.h
//
// CORA: contSet/@conPolyZono/exactPlus.m
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

ConPolyZono ConPolyZono::exact_plus(const ConPolyZono &S) const {
    detail::Entry entry;
    CORA_CHECK(S.dim() == dim(), Error::dimension_mismatch,
               [] { return std::string("exact_plus: the sets differ in dimension"); });
    CORA_CHECK(S.ids() == ids(), Error::wrong_value, [] {
        return std::string("exact_plus: both sets need the same identifiers, in the same order");
    });
    const ConPolyZono other(PolyZonotope(c(), generators(), independent_generators(), exponents(),
                                         ids(), detail::Verbatim{}),
                            S.cons_);
    CORA_CHECK(other.isequal(ConPolyZono(poly_, cons_), 0.0), Error::wrong_value, [] {
        return std::string("exact_plus: only defined for sets with identical constraints");
    });
    // the exponents of both on the same factors, side by side
    Exponents E = exponents();
    E.insert(E.end(), S.exponents().begin(), S.exponents().end());
    return {PolyZonotope(c() + S.c(), Tensor::concat({generators(), S.generators()}, 0),
                         Tensor::concat({independent_generators(), S.independent_generators()}, 0),
                         std::move(E), ids(), detail::Verbatim{}),
            cons_};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
