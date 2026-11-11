// split_dep_factor - the halves of a polynomial zonotope along one dependent factor
//
// The factor alpha in [-1, 1] is replaced by (alpha + 1) / 2 for the one half and (alpha - 1) / 2
// for the other, both again in [-1, 1]: a generator with the exponent i of the factor becomes
// i + 1 generators with the exponents 0, ..., i and the coefficients 0.5^i binom(i, m) (and
// (-1)^(i-m) for the second half). The generators that lose all their exponents move to the
// center. The halves cover the set; they are CORA's splitDepFactor with the default polyOrd (the
// highest exponent of the factor), whose extra factor does not occur then.
//
// Syntax:   [S1, S2] = S.split_dep_factor(id);
// Inputs:   id - the identifier of the factor, one of ids()
// Outputs:  S1, S2 - the halves, with the identifiers of S
// See also: split_longest_gen
//
// CORA: contSet/@polyZonotope/splitDepFactor.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "global/format.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// The columns -------------------------------------------------------------------------------------

/// The generators of one half, as columns of n entries, with their exponents.
struct Columns {
    std::vector<std::vector<double>> g;
    Exponents E;
};

/// The row i of Pascal's triangle: binom(i, 0), ..., binom(i, i).
std::vector<double> aux_pascal(int i) {
    std::vector<double> row = {1.0};
    for (int k = 0; k < i; ++k) {
        std::vector<double> next(row.size() + 1, 1.0);
        for (std::size_t m = 1; m < row.size(); ++m) next[m] = row[m - 1] + row[m];
        row = std::move(next);
    }
    return row;
}

/// The set from the columns: those without any exponent join the center, in order.
CORA_FROM_NUMBERS PolyZonotope aux_assemble(const PolyZonotope &base, const Columns &cols) {
    const int64_t n = base.center().shape()[0];
    std::vector<double> c = base.center().data();
    std::vector<std::vector<double>> g;
    Exponents E;
    for (std::size_t k = 0; k < cols.g.size(); ++k) {
        const std::vector<int> &e = cols.E[k];
        if (std::all_of(e.begin(), e.end(), [](int x) { return x == 0; })) {
            for (int64_t r = 0; r < n; ++r) c[std::size_t(r)] += cols.g[k][std::size_t(r)];
        } else {
            g.push_back(cols.g[k]);
            E.push_back(e);
        }
    }
    // the columns as an (n, h) matrix, row-major
    const auto h = int64_t(g.size());
    std::vector<double> G(std::size_t(n * h));
    for (int64_t j = 0; j < h; ++j)
        for (int64_t r = 0; r < n; ++r)
            G[std::size_t(r * h + j)] = g[std::size_t(j)][std::size_t(r)];
    return {Tensor::from_data(c, {n, 1}), Tensor::from_data(G, {n, h}),
            base.independent_generators(), std::move(E), base.ids()};
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The halves --------------------------------------------------------------------------------------

std::pair<PolyZonotope, PolyZonotope> PolyZonotope::split_dep_factor(int64_t id) const {
    detail::Entry entry;
    const auto it = std::find(id_.begin(), id_.end(), id);
    CORA_CHECK(it != id_.end(), Error::wrong_value, [&] {
        return "split_dep_factor: the identifier " + std::to_string(id) + " is not a factor";
    });
    const auto row = std::size_t(it - id_.begin());
    // the generators as columns of n numbers
    const int64_t n = dim(), h = num_dependent();
    const std::vector<double> G = G_.data();
    const auto column = [&](int64_t j) {
        std::vector<double> col(static_cast<std::size_t>(n));
        for (int64_t r = 0; r < n; ++r) col[std::size_t(r)] = G[std::size_t(r * h + j)];
        return col;
    };
    // the highest exponent of the factor
    int order = 0;
    for (const std::vector<int> &e : E_) order = std::max(order, e[row]);
    // the generators without the factor stay in both halves
    Columns first, second;
    for (int64_t j = 0; j < h; ++j)
        if (E_[std::size_t(j)][row] == 0) {
            first.g.push_back(column(j));
            first.E.push_back(E_[std::size_t(j)]);
        }
    second = first;

    // exponent i of the factor: the i + 1 powers of (alpha +- 1) / 2, block by block
    for (int i = 1; i <= order; ++i) {
        const std::vector<double> binom = aux_pascal(i);
        for (int m = 0; m <= i; ++m) {
            const double c1 = std::pow(0.5, i) * binom[std::size_t(m)];
            const double c2 = c1 * ((i - m) % 2 == 0 ? 1.0 : -1.0);
            for (int64_t j = 0; j < h; ++j) {
                if (E_[std::size_t(j)][row] != i) continue;
                // the generator scaled for each half, with the exponent m of the factor
                std::vector<double> g1 = column(j), g2 = g1;
                for (std::size_t r = 0; r < g1.size(); ++r) {
                    g1[r] *= c1;
                    g2[r] *= c2;
                }
                std::vector<int> e = E_[std::size_t(j)];
                e[row] = m;
                first.g.push_back(std::move(g1));
                second.g.push_back(std::move(g2));
                first.E.push_back(e);
                second.E.push_back(e);
            }
        }
    }
    // the generators without any exponent left have moved to the centers
    return {aux_assemble(*this, first), aux_assemble(*this, second)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
