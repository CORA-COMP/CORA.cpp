// cut - a constrained zonotope cut by a polytope, with the rows kept as rows on the factors
//
// CORA's polytope/and_ adds, for each inequality a'x <= b, a factor s and the equality
// a'(c + G x) + s r = m, with a linear program for r and m. Here the inequality stays one:
// a'G x <= b - a'c, widened by what carrying the row onto the factors rounds off (zero when up and
// down agree) and by the error box of the set (zero). An equality is its two halves, and no row is
// skipped for being redundant. The result is CORALean's `ConZonotope.intersectHPolytope`, bit for
// bit; boxing it by interval_tight keeps the rows, as the linear programs of CORA do.
//
// Syntax:   ConZonotope::Cut K = cz.cut(P);   ConZonotope::Cut K2 = K.linear_map(M);
// Inputs:   P - the polytope (halfspaces and equalities), M - a matrix (m, n)
// Outputs:  K - the set and its rows C x <= d on the factors
//           K2 - the image of K
// See also: intersect, interval_tight
//
// CORA.cpp only: CORALean's float definition replaces CORA's slack factors
// Authors:       Claude Sonnet 5.5
// Written:       10-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "cora/cont_set/polytope/polytope.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using Vec = std::vector<double>;

/// The dot product of a and the column j of the (n, h) matrix G, summed from the last term.
double aux_dot(const double *a, const Vec &G, int64_t n, int64_t h, int64_t j) {
    double acc = 0.0;
    for (int64_t l = n; l-- > 0;) acc = a[l] * G[std::size_t(l * h + j)] + acc;
    return acc;
}

/// The rows a x <= r of the polytope with its equalities as two halves, the rows of a first.
std::pair<Vec, Vec> aux_rows(const Polytope::Constraints &H) {
    const Vec A = priv_host_rows(H.A), b = priv_host_rows(H.b), Ae = priv_host_rows(H.Ae),
              be = priv_host_rows(H.be);
    Vec a = A, r;
    for (const double x : b) r.push_back(x + 0.0);
    a.insert(a.end(), Ae.begin(), Ae.end());
    for (const double x : be) r.push_back(x + 0.0);
    for (const double x : Ae) a.push_back(-x);
    for (const double x : be) r.push_back(-x + 0.0);
    return {a, r};
}

} // namespace


// ===========================================  MAIN  =========================================== //

ConZonotope::Cut ConZonotope::cut(const Polytope &P) const {
    detail::Entry entry;
    CORA_CHECK(P.dim() == dim(), Error::dimension_mismatch, [&] {
        return "cut: the sets differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(P.dim());
    });
    CORA_CHECK(c_.ndim() == 2, Error::not_supported,
               [] { return std::string("cut: a single set, not a batch"); });
    const int64_t n = dim(), h = G_.shape()[1];
    const auto [a, r] = aux_rows(P.constraints());
    const Vec c = c_.data(), G = G_.data();
    // each row onto the factors: a'G, and b - a'c widened by the carrying error and the error box
    Vec C, d;
    for (std::size_t i = 0; i < r.size(); ++i) {
        const double *row = a.data() + std::ptrdiff_t(i) * n;
        for (int64_t j = 0; j < h; ++j) C.push_back(aux_dot(row, G, n, h, j));
        const double slack = r[i] - aux_dot(row, c, n, 1, 0);
        d.push_back((slack + 0.0) + 0.0);
    }
    // the set with its rows (none: the empty blocks)
    const int64_t q = int64_t(r.size());
    return {c_,
            G_,
            A_,
            b_,
            q == 0 ? priv_zeros(c_, 0, h) : Tensor::like(c_, C, {q, h}),
            q == 0 ? priv_zeros(c_, 0, 1) : Tensor::like(c_, d, {q, 1})};
}

ConZonotope::Cut ConZonotope::Cut::linear_map(const Tensor &M) const {
    detail::Entry entry;
    CORA_CHECK(M.ndim() == 2 && M.shape()[1] == c.shape()[0], Error::dimension_mismatch, [&] {
        return "linear_map: the matrix needs " + std::to_string(c.shape()[0]) +
               " columns, got the shape " + shape_text(M.shape());
    });
    auto [c2, G2] = Tensor::zonotope_linear_map(M, c, G);
    return {std::move(c2), std::move(G2), A, b, C, d};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
