// priv_poly_power - a polynomial of one neuron evaluated on its polynomial zonotope
//
// See also: priv_poly_power.h
//
// CORA: nn/layers/nonlinear/@nnActivationLayer/evaluatePolyZonotopeNeuron.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       13-November-2026
// Last update:   ---
// Reviewed:      ---

#include "nn/neural_network/private/priv_poly_power.h"

#include "cora/global/tensor/tolerance.h"
#include "global/format.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using Row = std::vector<double>;

/// A range [begin, end) of a generator list.
struct Seg {
    int64_t begin = 0, end = 0;
    int64_t size() const { return end - begin; }
};

/// Where the powers sit: dep[i - 1] the new generators of the power i, dep_ext[i - 1] those of the
/// power i with all it is made of (CORA's G_start/G_end and G_ext_start/G_ext_end), ind and
/// ind_ext the same for the independent generators.
struct Layout {
    std::vector<Seg> dep, dep_ext, ind, ind_ext;
};

/// The ranges of the powers up to order for h dependent and q independent generators.
Layout aux_layout(int64_t h, int64_t q, int order) {
    Layout out;
    out.dep = out.dep_ext = {Seg{0, h}};
    out.ind = out.ind_ext = {Seg{0, q}};
    for (int o = 2; o <= order; ++o) {
        const std::size_t i1 = std::size_t(o / 2 - 1), i2 = std::size_t((o + 1) / 2 - 1);
        const bool equal = i1 == i2;
        // the generators of the two factors, and the new products
        const int64_t g1 = out.dep_ext[i1].size(), g2 = out.dep_ext[i2].size();
        const int64_t n1 = out.ind_ext[i1].size(), n2 = out.ind_ext[i2].size();
        const int64_t dep_new = equal ? g1 * (g1 + 1) / 2 : g1 * g2;
        const int64_t ind_new = equal ? n1 * (n1 + 1) / 2 + g1 * n2 : n1 * n2 + g1 * n2 + g2 * n1;
        out.dep.push_back({out.dep.back().end, out.dep.back().end + dep_new});
        out.dep_ext.push_back({out.dep_ext.back().end, out.dep_ext.back().end + g1 + g2 + dep_new});
        out.ind.push_back({out.ind.back().end, out.ind.back().end + ind_new});
        out.ind_ext.push_back({out.ind_ext.back().end, out.ind_ext.back().end + n1 + n2 + ind_new});
        if (out.dep_ext.back().end > numerics::kMaxPowerGenerators ||
            out.ind_ext.back().end > numerics::kMaxPowerGenerators)
            CORA_FAIL(Error::wrong_value, "evaluate: the powers of order " + std::to_string(order) +
                                              " of " + std::to_string(h) +
                                              " generators are too many");
    }
    return out;
}

/// The products of the entries of a with those of b (CORA's calcSquaredG): a fastest and then b;
/// of a list with itself (equal) the squares first, then twice each pair i < j.
Row aux_products(const Row &a, const Row &b, bool equal) {
    Row out;
    if (a.empty() || b.empty()) return out;
    if (equal) {
        const std::size_t n = a.size();
        for (std::size_t i = 0; i < n; ++i) out.push_back(a[i] * b[i]);
        for (std::size_t i = 0; i + 1 < n; ++i)
            for (std::size_t j = i + 1; j < n; ++j) out.push_back(2 * a[i] * b[j]);
        return out;
    }
    for (const double bj : b)
        for (const double ai : a) out.push_back(ai * bj);
    return out;
}

void aux_append(Row &out, const Row &part, double scale) {
    for (const double v : part) out.push_back(scale * v);
}

/// The set of a product of two powers (CORA's calcSquared): the centre and the new dependent
/// and independent generators; the square of an independent generator shifts the centre by half.
void aux_product(double c1, const Row &G1, const Row &GI1, double c2, const Row &G2, const Row &GI2,
                 bool equal, double &c, Row &G, Row &GI) {
    c = c1 * c2;
    G.clear();
    GI.clear();
    aux_append(G, G1, c2);
    aux_append(G, G2, c1);
    aux_append(G, aux_products(G1, G2, equal), 1.0);
    // the independent products: the squares shift the centre by half
    Row quad = aux_products(GI1, GI2, equal);
    if (equal) {
        for (std::size_t i = 0; i < GI1.size(); ++i) {
            quad[i] *= 0.5;
            c += quad[i];
        }
    }
    aux_append(GI, GI1, c2);
    aux_append(GI, GI2, c1);
    aux_append(GI, quad, 1.0);
    // the cross terms of the dependent and the independent generators, twice for a square
    if (equal) {
        for (const double v : aux_products(GI1, G2, false)) GI.push_back(2 * v);
        return;
    }
    aux_append(GI, aux_products(G1, GI2, false), 1.0);
    aux_append(GI, aux_products(GI1, G2, false), 1.0);
}

/// The columns [a_i + b_j] in the order of aux_products.
Exponents aux_exponent_sums(const Exponents &A, const Exponents &B, bool equal) {
    Exponents out;
    if (A.empty() || B.empty()) return out;
    const auto sum = [&](std::size_t i, std::size_t j) {
        std::vector<int> column = A[i];
        for (std::size_t r = 0; r < column.size(); ++r) column[r] += B[j][r];
        return column;
    };
    if (equal) {
        for (std::size_t i = 0; i < A.size(); ++i) out.push_back(sum(i, i));
        for (std::size_t i = 0; i + 1 < A.size(); ++i)
            for (std::size_t j = i + 1; j < A.size(); ++j) out.push_back(sum(i, j));
        return out;
    }
    for (std::size_t j = 0; j < B.size(); ++j)
        for (std::size_t i = 0; i < A.size(); ++i) out.push_back(sum(i, j));
    return out;
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The sizes ---------------------------------------------------------------------------------------

int64_t priv_power_dep_size(int64_t h, int order) { return aux_layout(h, 0, order).dep.back().end; }

int64_t priv_power_ind_size(int64_t h, int64_t q, int order) {
    return aux_layout(h, q, order).ind.back().end;
}

// The exponents -----------------------------------------------------------------------------------

Exponents priv_power_exponents(const Exponents &E, int order) {
    // the exponents of every power with all it is made of, the new ones last
    std::vector<Exponents> ext{E};
    Exponents out = E;
    for (int o = 2; o <= order; ++o) {
        const Exponents &E1 = ext[std::size_t(o / 2 - 1)], &E2 = ext[std::size_t((o + 1) / 2 - 1)];
        const Exponents added = aux_exponent_sums(E1, E2, o / 2 == (o + 1) / 2);
        Exponents joined = E1;
        joined.insert(joined.end(), E2.begin(), E2.end());
        joined.insert(joined.end(), added.begin(), added.end());
        out.insert(out.end(), added.begin(), added.end());
        ext.push_back(std::move(joined));
    }
    return out;
}

// The polynomial ----------------------------------------------------------------------------------

PrivNeuron priv_eval_poly(double c, const Row &G, const Row &GI, const Row &coeffs) {
    const int order = int(coeffs.size()) - 1;
    const Layout lay = aux_layout(int64_t(G.size()), int64_t(GI.size()), order);
    // the powers, each with the generators of its factors; the first is the set itself
    Row c_out(std::size_t(order), 0.0), G_ext(std::size_t(lay.dep_ext.back().end), 0.0);
    Row GI_ext(std::size_t(lay.ind_ext.back().end), 0.0);
    c_out[0] = c;
    std::copy(G.begin(), G.end(), G_ext.begin());
    std::copy(GI.begin(), GI.end(), GI_ext.begin());
    const auto slice = [](const Row &v, const Seg &s) {
        return Row(v.begin() + s.begin, v.begin() + s.end);
    };
    for (int i = 2; i <= order; ++i) {
        const std::size_t i1 = std::size_t(i / 2 - 1), i2 = std::size_t((i + 1) / 2 - 1);
        Row Gi, GIi;
        double ci = 0;
        aux_product(c_out[i1], slice(G_ext, lay.dep_ext[i1]), slice(GI_ext, lay.ind_ext[i1]),
                    c_out[i2], slice(G_ext, lay.dep_ext[i2]), slice(GI_ext, lay.ind_ext[i2]),
                    i1 == i2, ci, Gi, GIi);
        c_out[std::size_t(i - 1)] = ci;
        std::copy(Gi.begin(), Gi.end(), G_ext.begin() + lay.dep_ext[std::size_t(i - 1)].begin);
        std::copy(GIi.begin(), GIi.end(), GI_ext.begin() + lay.ind_ext[std::size_t(i - 1)].begin);
    }
    // the weights: the coefficient of x^i on the power i
    for (int i = 1; i <= order; ++i) {
        const double a = coeffs[std::size_t(order - i)];
        c_out[std::size_t(i - 1)] *= a;
        for (int64_t k = lay.dep_ext[std::size_t(i - 1)].begin;
             k < lay.dep_ext[std::size_t(i - 1)].end; ++k)
            G_ext[std::size_t(k)] *= a;
        for (int64_t k = lay.ind_ext[std::size_t(i - 1)].begin;
             k < lay.ind_ext[std::size_t(i - 1)].end; ++k)
            GI_ext[std::size_t(k)] *= a;
    }
    // the generators of the factors of a power add to the power's own, from the highest down
    PrivNeuron out;
    out.G.assign(std::size_t(lay.dep.back().end), 0.0);
    out.GI.assign(std::size_t(lay.ind.back().end), 0.0);
    const auto fold = [&](Row &ext, Row &own, const std::vector<Seg> &ext_seg,
                          const std::vector<Seg> &own_seg, int i) {
        const std::size_t i1 = std::size_t(i / 2 - 1), i2 = std::size_t((i + 1) / 2 - 1);
        int64_t at = ext_seg[std::size_t(i - 1)].begin;
        for (int64_t k = 0; k < ext_seg[i1].size(); ++k)
            ext[std::size_t(ext_seg[i1].begin + k)] += ext[std::size_t(at++)];
        for (int64_t k = 0; k < ext_seg[i2].size(); ++k)
            ext[std::size_t(ext_seg[i2].begin + k)] += ext[std::size_t(at++)];
        for (int64_t k = 0; k < own_seg[std::size_t(i - 1)].size(); ++k)
            own[std::size_t(own_seg[std::size_t(i - 1)].begin + k)] = ext[std::size_t(at++)];
    };
    for (int i = order; i >= 2; --i) {
        fold(G_ext, out.G, lay.dep_ext, lay.dep, i);
        fold(GI_ext, out.GI, lay.ind_ext, lay.ind, i);
    }
    // the first power, and the centre
    std::copy(G_ext.begin(), G_ext.begin() + lay.dep[0].end, out.G.begin());
    std::copy(GI_ext.begin(), GI_ext.begin() + lay.ind[0].end, out.GI.begin());
    for (const double v : c_out) out.c += v;
    out.c += coeffs.back();
    return out;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
