// intersect - the intersection of constrained polynomial zonotopes, as CORA's and_
//
// Both sets are made dependent (their independent generators on new factors, CORALean's
// dependent). The result is the first set on the factors of both, its generators padded by +0
// rows, with the rows of both constraints and the n agreement rows G1' a1 - G2' a2 = c2 - c1 that
// pin the two points together (CORALean's intersectionDep; -G2 as 0 - G2, so a zero stays +0).
//
// Syntax:   S = cPZ.intersect(cPZ2);   T = cPZ.dependent();
// Inputs:   cPZ2 - a constrained polynomial zonotope of the same dimension
// Outputs:  S - the intersection
//           T - the set without independent generators
// See also: con_poly_zono.h
//
// CORA: contSet/@conPolyZono/and_.m
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

ConPolyZono ConPolyZono::dependent() const {
    detail::Entry entry;
    const int64_t p = int64_t(ids().size()), q = independent_generators().shape()[0];
    const PolyZonotope P(c(), Tensor::concat({generators(), independent_generators()}, 0),
                         c().zeros_like({0, dim()}),
                         priv_block_exponents(exponents(), p, priv_identity_exponents(q), q),
                         priv_append_fresh_ids(ids(), q), detail::Verbatim{});
    return {P, cons_.pad_vars(q)};
}

ConPolyZono ConPolyZono::intersect(const ConPolyZono &S) const {
    detail::Entry entry;
    CORA_CHECK(S.dim() == dim(), Error::dimension_mismatch,
               [] { return std::string("intersect: the sets differ in dimension"); });
    const ConPolyZono a = dependent(), b = S.dependent();
    const int64_t p1 = int64_t(a.ids().size()), p2 = int64_t(b.ids().size());
    const int64_t h2 = b.generators().shape()[0];
    const Exponents E = priv_block_exponents(a.exponents(), p1, b.exponents(), p2);
    // the agreement rows: [G1', -G2'] (n, h1 + h2) = c2 - c1
    const Tensor rows =
        Tensor::concat({a.generators(), b.generators().zeros_like() - b.generators()}, 0)
            .matrix_transpose();
    const PolyLevelSet agree(rows, priv_plus_zero(b.c() - a.c()), E, p1 + p2);
    return priv_set(a.c(), Tensor::concat({a.generators(), c().zeros_like({h2, dim()})}, 0),
                    c().zeros_like({0, dim()}), E, a.cons_.cart_prod(b.cons_).and_rows(agree));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
