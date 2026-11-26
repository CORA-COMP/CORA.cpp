// convex_hull - the convex hull of constrained polynomial zonotopes, as CORA's convHull_
//
// Both sets are made dependent, then combined (linear_combination); the hull is n + 1 copies of
// that set, each on its own factors, weighted by w_j in [-1, 1] with sum_j w_j = 1 - (n + 1)
// (CORALean's convHullSingle): the center (n + 1) c summed from the last copy, the generators
// [c ... c | G ... G | G ... G], the weights' row last.
//
// Syntax:   S = cPZ.convex_hull(cPZ2);
// Inputs:   cPZ2 - a constrained polynomial zonotope of the same dimension
// Outputs:  S - a set that holds the convex hull
// See also: con_poly_zono.h
//
// CORA: contSet/@conPolyZono/convHull_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_poly_zono/private/priv.h"
#include "cora/global/error.h"
#include <cstdint>
#include <vector>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The exponents of convHullSingle: column i < a is weight i alone; block 2 and 3 hold copy j of
/// the columns E on factors j p ... j p + p - 1, block 3 also on weight j.
Exponents aux_exponents(const Exponents &E, int64_t p, int64_t a) {
    const int64_t h = int64_t(E.size()), rows = a * p + a;
    Exponents out(std::size_t(a + 2 * a * h), std::vector<int>(std::size_t(rows), 0));
    for (int64_t i = 0; i < a; ++i) out[std::size_t(i)][std::size_t(a * p + i)] = 1;
    for (int64_t j = 0; j < a; ++j)
        for (int64_t l = 0; l < h; ++l)
            for (int64_t k = 0; k < p; ++k) {
                const int e = E[std::size_t(l)][std::size_t(k)];
                out[std::size_t(a + j * h + l)][std::size_t(j * p + k)] = e;
                out[std::size_t(a + a * h + j * h + l)][std::size_t(j * p + k)] = e;
            }
    // the weight of copy j on the third block
    for (int64_t j = 0; j < a; ++j)
        for (int64_t l = 0; l < h; ++l)
            out[std::size_t(a + a * h + j * h + l)][std::size_t(a * p + j)] = 1;
    return out;
}

} // namespace


// ===========================================  MAIN  =========================================== //

ConPolyZono ConPolyZono::convex_hull(const ConPolyZono &S) const {
    detail::Entry entry;
    CORA_CHECK(S.dim() == dim(), Error::dimension_mismatch,
               [] { return std::string("convex_hull: the sets differ in dimension"); });
    const ConPolyZono X = dependent().linear_combination(S.dependent());
    const int64_t a = dim() + 1, p = int64_t(X.ids().size()), n = dim();
    // a c, summed from the last copy (CORALean's nsmulDown)
    Tensor c = X.c().zeros_like();
    for (int64_t j = 0; j < a; ++j) c = X.c() + c;
    std::vector<Tensor> rows(std::size_t(a), X.c().expand_dims(0));
    for (int64_t j = 0; j < 2 * a; ++j) rows.push_back(X.generators());
    // the copies' rows, then the weights' row sum_j w_j = 1 - a
    const PolyLevelSet row(c.ones_like({1, a}), c.full_like({1}, double(1 - a)),
                           priv_identity_exponents(a), a);
    return priv_set(c, Tensor::concat(rows, 0), c.zeros_like({0, n}),
                    aux_exponents(X.exponents(), p, a), X.cons_.copies(a).cart_prod(row));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
