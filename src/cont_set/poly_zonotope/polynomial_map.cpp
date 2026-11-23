// polynomial_map - the image of a polynomial zonotope under a polynomial, as CORA's polyMap
//
// The map z_i = sum_t coeff[i, t] prod_j x_j^E[j, t]. A term of exponent 0 goes to the constant.
// For the other terms the coordinates are multiplied one after the other (CORA's order: the factors
// by coordinate, lowest first) with CORALean's mul, over the set with its independent generators
// made dependent (absorbRounding), so every product is exact over the shared factors; the terms are
// then added by exact_plus. CORA multiplies pairs of coordinates by quad_map and keeps the
// independent part independent, which encloses it (docs/cora-differences.md). CORALean's evalPoly
// is the case of one variable per coordinate (Horner's rule) and rounds as mul does.
//
// Syntax:   S = pZ.polynomial_map(coeff, E);
// Inputs:   coeff - the coefficients (m, T), one column per term
//           E - the exponents of the terms: E[t] has one exponent per coordinate of the set
// Outputs:  S - the image, of dimension m
// See also: quad_map, cubic_map, exact_plus
//
// CORA: contSet/@polyZonotope/polyMap.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
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

/// The set with equal exponents merged and zero generators dropped (the constructor), constants in
/// the center.
PolyZonotope aux_tidy(const PolyZonotope &S) {
    return priv_fold_constants(
        {S.center(), S.generators(), S.independent_generators(), S.exponents(), S.ids()});
}

} // namespace


// ===========================================  MAIN  =========================================== //

PolyZonotope PolyZonotope::polynomial_map(const Tensor &coeff, const Exponents &E) const {
    detail::Entry entry;
    const int64_t n = dim();
    CORA_CHECK(coeff.ndim() == 2 && coeff.shape()[1] == int64_t(E.size()), Error::wrong_value, [&] {
        return "polynomial_map: coeff must have one column per term (" + std::to_string(E.size()) +
               "), got " + shape_text(coeff.shape());
    });
    for (const std::vector<int> &term : E) {
        CORA_CHECK(int64_t(term.size()) == n, Error::wrong_value, [&] {
            return "polynomial_map: every term needs one exponent per coordinate (" +
                   std::to_string(n) + ")";
        });
        CORA_CHECK(std::all_of(term.begin(), term.end(), [](int e) { return e >= 0; }),
                   Error::wrong_value,
                   [] { return std::string("polynomial_map: the exponents must be nonnegative"); });
    }
    const int64_t m = coeff.shape()[0];
    const PolyZonotope D = priv_absorb_rounding(*this);
    // the constant: the coefficients of the terms of exponent 0, added up
    std::vector<int64_t> constant, terms;
    for (std::size_t t = 0; t < E.size(); ++t)
        (std::all_of(E[t].begin(), E[t].end(), [](int e) { return e == 0; }) ? constant : terms)
            .push_back(int64_t(t));
    Tensor c = coeff.zeros_like({m});
    if (!constant.empty()) c = priv_rows(coeff.matrix_transpose(), constant).sum(-2);
    std::optional<PolyZonotope> result;
    for (const int64_t t : terms) {
        // the coordinates of the term, each as often as its exponent says
        std::optional<PolyZonotope> product;
        for (int64_t j = 0; j < n; ++j)
            for (int e = 0; e < E[std::size_t(t)][std::size_t(j)]; ++e) {
                const PolyZonotope xj = D.project({j}).as<PolyZonotope>();
                // the product, its equal exponents merged and its constants in the center
                product = product ? aux_tidy(priv_multiply(*product, xj)) : xj;
            }
        const PolyZonotope term = product->linear_map(coeff.slice(-1, t, t + 1));
        result = result ? result->exact_plus(term) : term;
    }
    if (!result) return {c, G_.zeros_like({0, m}), G_.zeros_like({0, m}), {}, id_};
    return result->plus(c);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
