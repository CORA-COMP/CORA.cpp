// plus - the Minkowski sum of a constrained polynomial zonotope, as CORA's plus
//
// With a constrained polynomial zonotope the centers add and both sets keep their factors, side by
// side: the generators and the constraints join on blocks (CORALean's plus), the identifiers are
// 1..p as CORA sets them. A zonotope or a point joins the independent part (CORA's plus).
//
// Syntax:   S = cPZ.plus(S2);   S = cPZ + Z;   S = cPZ + v;
// Inputs:   S2 - a set of the same dimension, or a point v (n)
// Outputs:  S - the sum
// See also: con_poly_zono.h
//
// CORA: contSet/@conPolyZono/plus.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_poly_zono/private/priv.h"
#include "cora/global/error.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The dimensions agree, else Error::dimension_mismatch.
void aux_same_dim(int64_t n, int64_t m) {
    detail::Entry entry;
    CORA_CHECK(n == m, Error::dimension_mismatch, [&] {
        return "plus: the sets differ in dimension, " + std::to_string(n) + " and " +
               std::to_string(m);
    });
}

} // namespace


// ===========================================  MAIN  =========================================== //

ConPolyZono ConPolyZono::plus(const ConPolyZono &S) const {
    detail::Entry entry;
    aux_same_dim(dim(), S.dim());
    const int64_t p1 = int64_t(ids().size()), p2 = int64_t(S.ids().size());
    return priv_set(c() + S.c(), Tensor::concat({generators(), S.generators()}, 0),
                    Tensor::concat({independent_generators(), S.independent_generators()}, 0),
                    priv_block_exponents(exponents(), p1, S.exponents(), p2),
                    cons_.cart_prod(S.cons_));
}

ConPolyZono ConPolyZono::plus(const Zonotope &Z) const {
    detail::Entry entry;
    // the zonotope joins the independent part
    aux_same_dim(dim(), Z.dim());
    const PolyZonotope P(c() + Z.center(), generators(),
                         Tensor::concat({independent_generators(), Z.generators()}, 0), exponents(),
                         ids(), detail::Verbatim{});
    return {P, cons_};
}

ConPolyZono ConPolyZono::plus(const Tensor &v) const {
    detail::Entry entry;
    CORA_CHECK(v.ndim() == 1 && v.shape()[0] == dim(), Error::dimension_mismatch, [&] {
        return "plus: the point needs " + std::to_string(dim()) + " entries, got the shape " +
               shape_text(v.shape());
    });
    // a point moves the center
    return {PolyZonotope(c() + v, generators(), independent_generators(), exponents(), ids(),
                         detail::Verbatim{}),
            cons_};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
