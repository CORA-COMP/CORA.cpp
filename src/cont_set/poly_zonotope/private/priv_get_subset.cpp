// priv_get_subset - CORALean's getSubset of a block of generators, and its block exponents
//
// Factor k becomes cen + rad x_k in every column: (cen + rad x)^e = sum_j C(e, j) cen^(e-j) rad^j
// x^j, one column per j = 0, ..., max_e (a term with j > e is zero), the generator times
// C(e, j) cen^(e-j), then times rad^j. The powers are repeated products (CORALean's powDown), no
// column is merged or dropped. Used for the generators and the constraint rows of ConPolyZono.
//
// Syntax:   ExpandedColumns x = priv_get_subset_expand(G, E, k, cen, rad);
//           Exponents E = priv_block_exponents(E1, p1, E2, p2);
// See also: priv.h, get_subset.cpp
//
// CORA.cpp only: CORALean's PolyZonotope.getSubset, the binomial terms kept apart
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"

#include <algorithm>
#include <cstdint>
#include <utility>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// a^k by repeated products from 1 (CORALean's powDown at nearest).
double aux_power(double a, int k) {
    double r = 1.0;
    for (int t = 0; t < k; ++t) r = a * r;
    return r;
}

/// The binomial coefficient C(e, j), 0 for j > e, exact for the exponents of a set.
double aux_choose(int e, int j) {
    if (j > e) return 0.0;
    double r = 1.0;
    for (int t = 1; t <= j; ++t) r = r * double(e - j + t) / double(t);
    return r;
}

} // namespace


// ===========================================  MAIN  =========================================== //

ExpandedColumns priv_get_subset_expand(const Tensor &G, const Exponents &E, std::size_t k,
                                       double cen, double rad) {
    const int64_t h = int64_t(E.size()), n = G.shape()[1];
    int e_max = 0;
    for (const std::vector<int> &col : E) e_max = std::max(e_max, col[k]);
    const int64_t w = e_max + 1;
    const std::vector<double> g = G.data();
    std::vector<double> out(std::size_t(h * w * n));
    Exponents Ex;
    Ex.reserve(std::size_t(h * w));
    for (int64_t i = 0; i < h; ++i) {
        const int e = E[std::size_t(i)][k];
        for (int j = 0; j <= e_max; ++j) {
            // the coefficient C(e, j) cen^(e-j), then times rad^j, then times the generator
            const double f =
                aux_choose(e, j) * aux_power(cen, std::max(e - j, 0)) * aux_power(rad, j);
            const int64_t col = i * w + j;
            for (int64_t r = 0; r < n; ++r)
                out[std::size_t(col * n + r)] = f * g[std::size_t(i * n + r)] + 0.0;
            Ex.push_back(E[std::size_t(i)]);
            Ex.back()[k] = j;
        }
    }
    return {Tensor::like(G, out, {h * w, n}), std::move(Ex)};
}

Exponents priv_block_exponents(const Exponents &E1, int64_t p1, const Exponents &E2, int64_t p2) {
    Exponents out;
    out.reserve(E1.size() + E2.size());
    // the first block's columns, zero on the second block's factors
    for (const std::vector<int> &col : E1) {
        out.push_back(col);
        out.back().resize(std::size_t(p1 + p2), 0);
    }
    // the second block's, zero on the first's
    for (const std::vector<int> &col : E2) {
        std::vector<int> e(std::size_t(p1), 0);
        e.insert(e.end(), col.begin(), col.end());
        out.push_back(std::move(e));
    }
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
