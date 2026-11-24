// outer_approx - the small generators boxed, as CORA's outerApprox
//
// The generators that depend on a factor outside ids are candidates, together with the independent
// ones; those of only even exponents are halved (their monomial lies in [0, 1]). The candidates are
// sorted by the sum of their magnitudes and the shortest ones are taken while their running sum
// stays within tol times the largest magnitude of the coordinate. The dependent ones taken leave
// the set, and a box of all those taken (each coordinate the sum of the magnitudes, added from the
// last generator to the first) is appended to the independent generators. As CORALean's boxOut,
// the independent generators taken stay as they are.
//
// Syntax:   S = pZ.outer_approx(tol, ids);
// Inputs:   tol - relative size below which generators are boxed
//           ids - identifiers of the factors that stay (a generator in them alone is no candidate)
// Outputs:  S - the set with the dependent generators taken replaced by a box
// See also: reduce, restructure
//
// CORA: contSet/@polyZonotope/outerApprox.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "global/format.h"

#include <algorithm>
#include <cmath>
#include <numeric>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The sum of the entries of every row of A (r, n), added from the last entry to the first.
Tensor aux_row_sums(const Tensor &A) {
    Tensor sum = A.slice(-1, 0, 1).zeros_like();
    for (int64_t k = A.shape()[1]; k-- > 0;) sum = A.slice(-1, k, k + 1) + sum;
    return sum.squeeze(-1);
}

/// The sum of the rows of A (r, n), added from the last row to the first.
Tensor aux_column_sums(const Tensor &A) {
    Tensor sum = A.zeros_like({A.shape()[1]});
    for (int64_t r = A.shape()[0]; r-- > 0;) sum = A.slice(-2, r, r + 1).squeeze(-2) + sum;
    return sum;
}

} // namespace


// ===========================================  MAIN  =========================================== //

PolyZonotope PolyZonotope::outer_approx(double tol, const std::vector<int64_t> &ids) const {
    detail::Entry entry;
    CORA_CHECK(tol >= 0.0 && !std::isnan(tol), Error::wrong_value,
               [] { return std::string("outer_approx: tol must be nonnegative"); });
    const int64_t n = dim(), h = num_dependent(), q = GI_.shape()[0];
    // The candidates ---------------------------------------------------------------------------
    // the candidates: dependent generators with a factor outside ids, then the independent ones
    std::vector<bool> kept(id_.size(), false);
    for (const int64_t id : ids) {
        const int64_t at = priv_find_id(id_, id);
        if (at >= 0) kept[std::size_t(at)] = true;
    }
    std::vector<int64_t> dependent;
    for (int64_t i = 0; i < h; ++i)
        for (std::size_t k = 0; k < id_.size(); ++k)
            if (!kept[k] && E_[std::size_t(i)][k] > 0) {
                dependent.push_back(i);
                break;
            }
    // the even ones are halved: their monomial lies in [0, 1]
    std::vector<double> halve;
    for (const int64_t i : dependent) {
        const std::vector<int> &col = E_[std::size_t(i)];
        const bool even = std::all_of(col.begin(), col.end(), [](int e) { return e % 2 == 0; });
        halve.push_back(even ? 0.5 : 1.0);
    }
    const auto nd = int64_t(dependent.size());
    const Tensor G = priv_rows(G_, dependent).multiply(Tensor::like(G_, halve, {nd, 1}));
    const Tensor GGI = Tensor::concat({G, GI_}, -2);
    // The scan ---------------------------------------------------------------------------------
    // the candidates by the sum of their magnitudes, shortest first, the order of equals kept
    const int64_t total = nd + q;
    const std::vector<double> size = aux_row_sums(GGI.abs()).data();
    std::vector<int64_t> order(static_cast<std::size_t>(total));
    std::iota(order.begin(), order.end(), int64_t(0));
    std::sort(order.begin(), order.end(), [&](int64_t a, int64_t b) {
        return size[std::size_t(a)] != size[std::size_t(b)]
                   ? size[std::size_t(a)] < size[std::size_t(b)]
                   : a < b;
    });
    // the threshold per coordinate, and the scan: take while the running sum stays within it
    std::vector<double> abs_all = GGI.abs().data(), threshold(std::size_t(n), 0.0),
                        running(std::size_t(n), 0.0);
    for (int64_t j = 0; j < total; ++j)
        for (int64_t k = 0; k < n; ++k)
            threshold[std::size_t(k)] =
                std::max(threshold[std::size_t(k)], abs_all[std::size_t(j * n + k)]);
    for (double &t : threshold) t *= tol;
    int64_t count = 0;
    for (; count < total; ++count) {
        const int64_t j = order[std::size_t(count)];
        bool within = true;
        for (int64_t k = 0; k < n; ++k)
            within = within && abs_all[std::size_t(j * n + k)] + running[std::size_t(k)] <=
                                   threshold[std::size_t(k)];
        if (!within) break;
        for (int64_t k = 0; k < n; ++k)
            running[std::size_t(k)] = abs_all[std::size_t(j * n + k)] + running[std::size_t(k)];
    }
    // The result -------------------------------------------------------------------------------
    // the taken generators: dependent ones leave, all of them go into the box
    std::vector<bool> removed(std::size_t(h), false), taken(std::size_t(q), false);
    for (int64_t r = 0; r < count; ++r) {
        const int64_t j = order[std::size_t(r)];
        if (j < nd)
            removed[std::size_t(dependent[std::size_t(j)])] = true;
        else
            taken[std::size_t(j - nd)] = true;
    }
    std::vector<double> mask_g, mask_gi;
    std::vector<int64_t> stay;
    for (int64_t i = 0; i < h; ++i) {
        mask_g.push_back(removed[std::size_t(i)] ? 1.0 : 0.0);
        if (!removed[std::size_t(i)]) stay.push_back(i);
    }
    for (int64_t j = 0; j < q; ++j) mask_gi.push_back(taken[std::size_t(j)] ? 1.0 : 0.0);
    // the radius of the box: the removed magnitudes plus the taken ones, each from the last row
    const Tensor mg = Tensor::like(G_, mask_g, {h, 1}), mgi = Tensor::like(G_, mask_gi, {q, 1});
    const Tensor radius =
        aux_column_sums(G_.abs().multiply(mg)) + aux_column_sums(GI_.abs().multiply(mgi));
    // the generators that stay, and the box as the diagonal of the independent block
    Exponents E;
    for (const int64_t i : stay) E.push_back(E_[std::size_t(i)]);
    return {c_,
            priv_rows(G_, stay),
            Tensor::concat({GI_, radius.expand_dims(-1).diag()}, -2),
            std::move(E),
            id_,
            detail::Verbatim{}};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
