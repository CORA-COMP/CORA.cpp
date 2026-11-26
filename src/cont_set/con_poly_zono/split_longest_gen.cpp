// split_longest_gen - the halves along the longest factor, as CORA's splitLongestGen
//
// Each factor is measured by the summed squared lengths of the generators that use it (all-even
// columns halved; the columns of a factor that appears alone and linearly left out unless all are
// such, CORALean's longestFactor); the first factor of the largest measure is split. A set
// without factors is returned twice.
//
// Syntax:   [S1, S2] = cPZ.split_longest_gen();
// Inputs:   -
// Outputs:  S1, S2 - the halves
// See also: con_poly_zono.h
//
// CORA: contSet/@conPolyZono/splitLongestGen.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_poly_zono/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether column j is factor i alone, to the power 1, and factor i appears nowhere else.
bool aux_alone(const Exponents &E, std::size_t i, std::size_t j) {
    int total = 0;
    for (const std::vector<int> &col : E) total += col[i];
    if (total != 1 || E[j][i] != 1) return false;
    for (std::size_t k = 0; k < E[j].size(); ++k)
        if (k != i && E[j][k] != 0) return false;
    return true;
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::pair<ConPolyZono, ConPolyZono> ConPolyZono::split_longest_gen() const {
    detail::Entry entry;
    const Exponents &E = exponents();
    const std::size_t p = ids().size(), h = E.size(), n = std::size_t(dim());
    if (p == 0) return {*this, *this};
    // the columns that count: all but the lone linear ones, unless every column is one
    std::vector<bool> alone(h, false);
    std::size_t count = 0;
    for (std::size_t j = 0; j < h; ++j)
        for (std::size_t i = 0; i < p && !alone[j]; ++i)
            if (aux_alone(E, i, j)) alone[j] = true;
    for (const bool a : alone) count += a ? 1 : 0;
    const bool all = count >= h;
    // the squared length of each column, an all-even one halved first
    const std::vector<double> g = generators().data();
    std::vector<double> len(h);
    for (std::size_t j = 0; j < h; ++j) {
        bool even = true;
        for (const int e : E[j]) even = even && e % 2 == 0;
        double s = 0;
        for (std::size_t r = n; r-- > 0;) {
            const double v = even ? g[j * n + r] * 0.5 : g[j * n + r];
            s = v * v + s;
        }
        len[j] = s;
    }
    // the measure of each factor, and the first of the largest
    std::size_t best = 0;
    double best_len = 0;
    for (std::size_t i = 0; i < p; ++i) {
        double s = 0;
        for (std::size_t j = h; j-- > 0;)
            s = ((all || !alone[j]) && E[j][i] > 0 ? len[j] : 0.0) + s;
        if (i == 0 || !(s <= best_len)) {
            best = i;
            best_len = s;
        }
    }
    return split_dep_factor(ids()[best]);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
