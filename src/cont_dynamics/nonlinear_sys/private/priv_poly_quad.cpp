// priv_poly_quad - the quadratic maps of poly-adaptive, as CORALean's float layer writes them
//
// CORALean's PolyZonotope.quadMap is the bilinear map of the set with itself after absorbRounding:
// all (h + 1)^2 products of the columns [c G] of the set, each pair (i, j) apart (no merge of the
// pairs (i, j) and (j, i)), at the exponent column E_i + E_j, and each product the rounded dot
// product of column i with Q times column j. quadMapMixed pairs two sets over disjoint factors the
// same way, reading the first against the second. The second-order error of a window is the
// zonotope of the three products with the difference (zonotopeEven): an all-even column keeps half
// of itself and shifts the other half into the center, a constant column moves whole into the
// center. Structurally zero columns (the n generators absorbRounding adds for an empty error box,
// the pairs of a mixed product with a zero block) are left out; they rank first and are boxed at
// radius 0 in every reduction after, so nothing downstream reads them.
//
// Syntax:   PolyZonotope P = priv_poly_quad_map(Z, Q);
//           SecondOrder S = priv_second_order_error(Zdelta, Zdiff, Q);
// Inputs:   Z - polynomial zonotope, Zdelta, Zdiff - zonotopes, Q - the half Hessians (n, d, d)
// Outputs:  the image, and the generators and the center shift of the error
// See also: priv_abstrerr_poly_adaptive.cpp, quad_map.cpp
//
// CORA: contSet/@polyZonotope/quadMap.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_dynamics/nonlinear_sys/private/priv_poly_adaptive.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

// The products ------------------------------------------------------------------------------------

namespace {

/// The columns of a set as vectors: the center first, then the generators.
std::vector<Vec> aux_columns(const Tensor &c, const Tensor &G) {
    const int64_t d = c.shape()[0], N = G.shape().back();
    const Vec cv = c.data(), gv = G.data();
    std::vector<Vec> out(std::size_t(N + 1), Vec(std::size_t(d)));
    for (int64_t i = 0; i < d; ++i) {
        out[0][std::size_t(i)] = cv[std::size_t(i)];
        for (int64_t j = 0; j < N; ++j)
            out[std::size_t(j + 1)][std::size_t(i)] = gv[std::size_t(i * N + j)];
    }
    return out;
}

/// Q applied to a column, row by row (CORALean's mulVecDown at nearest: every dot product from the
/// last term to the first), for each of the m matrices.
struct QB {
    std::vector<Vec> w; // w[k][r] = (Q_k b)_r
};

QB aux_apply(const std::vector<Vec> &Q, int64_t d, const Vec &b) {
    QB out;
    for (const Vec &Qk : Q) {
        Vec w(static_cast<std::size_t>(d));
        for (int64_t r = 0; r < d; ++r) {
            double sum = 0;
            for (int64_t c = d; c-- > 0;)
                sum = Qk[std::size_t(r * d + c)] * b[std::size_t(c)] + sum;
            w[std::size_t(r)] = sum;
        }
        out.w.push_back(std::move(w));
    }
    return out;
}

/// The k-th entry of the product a^T Q b with Q b = w (CORALean's quadDown at nearest).
double aux_dot(const Vec &a, const Vec &w) {
    double sum = 0;
    for (std::size_t r = a.size(); r-- > 0;) sum = a[r] * w[r] + sum;
    return sum;
}

std::vector<Vec> aux_matrices(const std::vector<Tensor> &Q) {
    std::vector<Vec> out;
    for (const Tensor &Qk : Q) out.push_back(Qk.data());
    return out;
}

/// The products of the columns a_i (rows) against b_j (columns), all pairs i-major, as columns of m
/// entries.
std::vector<Vec> aux_products(const std::vector<Vec> &Q, int64_t d, const std::vector<Vec> &A,
                              const std::vector<Vec> &B) {
    // Q applied to every column of B once, then the dot products with the columns of A
    std::vector<QB> qb;
    for (const Vec &b : B) qb.push_back(aux_apply(Q, d, b));
    std::vector<Vec> out;
    for (const Vec &a : A)
        for (const QB &w : qb) {
            Vec col(Q.size());
            for (std::size_t k = 0; k < Q.size(); ++k) col[k] = aux_dot(a, w.w[k]);
            out.push_back(std::move(col));
        }
    return out;
}

Tensor aux_matrix(const Tensor &like, const std::vector<Vec> &cols, int64_t m) {
    const auto K = int64_t(cols.size());
    if (K == 0) return like.zeros_like({m, 0});
    Vec data(std::size_t(m * K));
    for (int64_t j = 0; j < K; ++j)
        for (int64_t i = 0; i < m; ++i)
            data[std::size_t(i * K + j)] = cols[std::size_t(j)][std::size_t(i)];
    return Tensor::like(like, data, {m, K});
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The quadratic maps ------------------------------------------------------------------------------

PolyZonotope priv_poly_quad_map(const PolyZonotope &Z, const std::vector<Tensor> &Q) {
    const int64_t d = Z.dim(), m = int64_t(Q.size());
    // the set with its independent generators made dependent, and the products of its columns
    const PolyZonotope A = priv_absorb_independent(Z);
    const std::vector<Vec> cols = aux_columns(A.center(), A.generators());
    const std::vector<Vec> prod = aux_products(aux_matrices(Q), d, cols, cols);
    // the exponent column of the pair: the two added, the center's being zero
    Exponents Eext = A.exponents();
    Eext.insert(Eext.begin(), std::vector<int>(A.ids().size(), 0));
    // the pairs in the order of the products
    Exponents E;
    for (const std::vector<int> &ei : Eext)
        for (const std::vector<int> &ej : Eext) {
            std::vector<int> col = ei;
            for (std::size_t k = 0; k < col.size(); ++k) col[k] += ej[k];
            E.push_back(std::move(col));
        }
    // the image: center 0, no independent generators
    const Tensor like = Z.center();
    return {like.zeros_like({m, 1}),
            aux_matrix(like, prod, m),
            like.zeros_like({m, 0}),
            std::move(E),
            A.ids(),
            detail::Verbatim{}};
}

SecondOrder priv_second_order_error(const Zonotope &Zdelta, const Zonotope &Zdiff,
                                    const std::vector<Tensor> &Q) {
    const int64_t d = Zdelta.dim(), m = int64_t(Q.size());
    const std::vector<Vec> Qv = aux_matrices(Q);
    const std::vector<Vec> cd = aux_columns(Zdelta.center(), Zdelta.generators());
    const std::vector<Vec> cf = aux_columns(Zdiff.center(), Zdiff.generators());
    std::vector<Vec> gens;  // the generators kept whole, in column order
    std::vector<Vec> shift; // the shifts into the center, in column order
    // the two mixed products: every column is odd except the constant one
    for (int side = 0; side < 2; ++side) {
        const std::vector<Vec> prod =
            side == 0 ? aux_products(Qv, d, cd, cf) : aux_products(Qv, d, cf, cd);
        shift.push_back(prod.front());
        for (std::size_t l = 1; l < prod.size(); ++l) gens.push_back(prod[l]);
    }
    // the quadratic map of the difference: the squares keep half, the constant moves whole
    const std::vector<Vec> prod = aux_products(Qv, d, cf, cf);
    const std::size_t N = cf.size();
    shift.push_back(prod.front());
    for (std::size_t i = 0; i < N; ++i)
        for (std::size_t j = 0; j < N; ++j) {
            if (i == 0 && j == 0) continue;
            Vec col = prod[i * N + j];
            if (i == j) {
                for (double &v : col) v = 0.5 * v;
                shift.push_back(col);
            }
            gens.push_back(std::move(col));
        }
    // the shift, from the last column to the first
    Vec s(static_cast<std::size_t>(m), 0.0);
    for (std::size_t l = shift.size(); l-- > 0;)
        for (std::size_t i = 0; i < s.size(); ++i) s[i] = shift[l][i] + s[i];
    return {aux_matrix(Zdelta.center(), gens, m), s};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
