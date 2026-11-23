// subs - factors of a polynomial zonotope replaced by polynomials, as CORA's subs
//
// The factor ids[k] becomes the k-th coordinate of inner, a polynomial in the factors of inner:
// every dependent generator is multiplied out over the powers of these polynomials, as CORALean's
// subs: the powers and the product over the coordinates are built term by term (the first factor
// the slower index, equal exponents not merged), each term scales the generator once; equal
// exponents merge only in the result. Factors of inner that also belong to this set are the same
// factors. CORA takes a shortcut when each coordinate of inner is a single monomial and merges the
// terms of every product; the set is the same, the rounding differs (docs/cora-differences.md).
//
// Syntax:   S = pZ.subs(inner);   S = pZ.subs(inner, ids);
// Inputs:   inner - a polynomial zonotope without independent generators, one coordinate per factor
//           ids - the identifiers of pZ that are replaced (default: all of them, in order)
// Outputs:  S - over the remaining factors of pZ followed by the factors of inner it did not have
// See also: resolve, get_subset
//
// CORA: contSet/@polyZonotope/subs.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "global/format.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The polynomials ---------------------------------------------------------------------------------

/// A polynomial in the factors of inner: a coefficient per term (m, 1) and the exponents of each.
struct Poly {
    Tensor coef;
    Exponents E;
};

// The product -------------------------------------------------------------------------------------

/// CORALean's subsMul: one term per pair, those of a the slower, with the products of the
/// coefficients and the sums of the exponents; equal exponents are not merged, so the terms are the
/// ones the float definition rounds.
Poly aux_multiply(const Poly &a, const Poly &b) {
    const std::size_t m1 = a.E.size(), m2 = b.E.size();
    Exponents E;
    for (std::size_t j1 = 0; j1 < m1; ++j1)
        for (std::size_t j2 = 0; j2 < m2; ++j2) {
            std::vector<int> col = a.E[j1];
            for (std::size_t i = 0; i < col.size(); ++i) col[i] += b.E[j2][i];
            E.push_back(std::move(col));
        }
    return {a.coef.multiply(b.coef.matrix_transpose()).reshape({int64_t(m1 * m2), 1}),
            std::move(E)};
}

} // namespace


// ===========================================  MAIN  =========================================== //

PolyZonotope PolyZonotope::subs(const PolyZonotope &inner,
                                const std::optional<std::vector<int64_t>> &ids) const {
    detail::Entry entry;
    const std::vector<int64_t> idu = ids ? *ids : id_;
    const int64_t nu = inner.dim();
    CORA_CHECK(int64_t(idu.size()) == nu &&
                   std::all_of(idu.begin(), idu.end(),
                               [&](int64_t v) { return priv_find_id(id_, v) >= 0; }),
               Error::wrong_value, [] {
                   return std::string("subs: the number of coordinates of inner must match the "
                                      "number of factors ids");
               });
    CORA_CHECK(inner.independent_generators().shape()[0] == 0 ||
                   inner.independent_generators().abs().max().item() == 0.0,
               Error::wrong_value,
               [] { return std::string("subs: inner must not have independent generators"); });
    if (num_dependent() == 0) return *this;
    // the factors that stay and the factors of inner that are new, then the union of both
    std::vector<int64_t> idr, idp;
    for (const int64_t v : id_)
        if (std::find(idu.begin(), idu.end(), v) == idu.end()) idr.push_back(v);
    for (const int64_t v : inner.ids())
        if (priv_find_id(idr, v) < 0) idp.push_back(v);
    std::vector<int64_t> id = idr;
    id.insert(id.end(), idp.begin(), idp.end());
    // the polynomial of each substituted factor: constant term, then the terms of inner
    const std::size_t pin = inner.ids().size();
    const std::size_t count = std::size_t(nu);
    std::vector<std::vector<Poly>> powers(count);
    int d = 0;
    for (const int64_t u : idu) {
        const std::size_t at = std::size_t(priv_find_id(id_, u));
        for (const std::vector<int> &col : E_) d = std::max(d, col[at]);
    }
    for (int64_t k = 0; k < nu; ++k) {
        const Tensor row = inner.generators().slice(-1, k, k + 1);
        Exponents E = {std::vector<int>(pin, 0)};
        E.insert(E.end(), inner.exponents().begin(), inner.exponents().end());
        const Poly p{Tensor::concat({inner.center().slice(-1, k, k + 1).reshape({1, 1}), row}, -2),
                     std::move(E)};
        powers[std::size_t(k)].push_back({row.ones_like({1, 1}), {std::vector<int>(pin, 0)}});
        powers[std::size_t(k)].push_back(p);
        for (int e = 2; e <= d; ++e)
            powers[std::size_t(k)].push_back(aux_multiply(p, powers[std::size_t(k)].back()));
    }
    // each generator: the product of the powers, scaled by the generator, and its exponents
    std::vector<Tensor> parts;
    Exponents E_all;
    for (std::size_t j = 0; j < E_.size(); ++j) {
        Poly product = powers[0][std::size_t(E_[j][std::size_t(priv_find_id(id_, idu[0]))])];
        for (std::size_t k = 1; k < idu.size(); ++k)
            product = aux_multiply(
                product, powers[k][std::size_t(E_[j][std::size_t(priv_find_id(id_, idu[k]))])]);
        parts.push_back(product.coef.multiply(G_.slice(-2, int64_t(j), int64_t(j) + 1)));
        // the exponents of each term over the remaining factors and those of the inner set
        for (const std::vector<int> &term : product.E) {
            std::vector<int> col(id.size(), 0);
            for (std::size_t i = 0; i < idr.size(); ++i)
                col[i] = E_[j][std::size_t(priv_find_id(id_, idr[i]))];
            for (std::size_t i = 0; i < pin; ++i)
                col[std::size_t(priv_find_id(id, inner.ids()[i]))] += term[i];
            E_all.push_back(std::move(col));
        }
    }
    // equal exponents merge, the constants join the center
    MergedColumns merged = priv_merge_columns(Tensor::concat(parts, -2), E_all);
    const PolyZonotope folded = priv_fold_constants(
        PolyZonotope(c_, std::move(merged.G), GI_, std::move(merged.E), id, detail::Verbatim{}));
    return {folded.center(),    folded.generators(), GI_,
            folded.exponents(), std::move(id),       detail::Verbatim{}};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
