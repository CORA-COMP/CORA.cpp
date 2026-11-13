// priv_poly_layer - the layer of order k on a polynomial zonotope, neuron by neuron
//
// See also: priv_layers.h
//
// CORA: nn/layers/nonlinear/@nnActivationLayer/evaluatePolyZonotope.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Claude Sonnet 5.5
// Written:       13-November-2026
// Last update:   ---
// Reviewed:      ---

#include "nn/neural_network/private/priv_activation_poly.h"
#include "nn/neural_network/private/priv_layers.h"
#include "nn/neural_network/private/priv_poly_power.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <numeric>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

using Column = std::vector<double>;

/// The rows on the host: the generators as columns of n entries.
struct Host {
    int64_t n = 0;
    Column c;
    std::vector<Column> G, GI;
    Exponents E;
    std::vector<int64_t> ids, gi_src;
    int64_t last_id = 0;
};

// The host rows -----------------------------------------------------------------------------------

/// The columns of an (n, k) tensor.
std::vector<Column> aux_columns(const Tensor &T) {
    const int64_t n = T.shape()[0], k = T.shape()[1];
    const std::vector<double> data = T.data();
    const Column zero(std::size_t(n), 0.0);
    std::vector<Column> out(std::size_t(k), zero);
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < k; ++j)
            out[std::size_t(j)][std::size_t(i)] = data[std::size_t(i * k + j)];
    return out;
}

/// The tensor (n, k) of columns, made like `like`.
Tensor aux_tensor(const Tensor &like, const std::vector<Column> &columns, int64_t n) {
    // row-major, as the tensor
    const int64_t k = int64_t(columns.size());
    std::vector<double> data(std::size_t(n * k));
    for (int64_t i = 0; i < n; ++i)
        for (int64_t j = 0; j < k; ++j)
            data[std::size_t(i * k + j)] = columns[std::size_t(j)][std::size_t(i)];
    return Tensor::like(like, data, {n, k});
}

Host aux_load(const PrivRows &r) {
    // the tensors to the host, the exponents and identifiers as they are
    Host h;
    h.n = r.c.shape()[0];
    h.c = r.c.data();
    h.G = aux_columns(r.G);
    h.GI = aux_columns(r.GI);
    h.E = r.E;
    h.ids = r.ids;
    h.gi_src = r.gi_src;
    h.last_id = r.last_id;
    return h;
}

void aux_store(PrivRows &r, const Host &h) {
    // the host rows back into tensors made like the centre
    r.c = Tensor::like(r.c, h.c, {h.n, 1});
    r.G = aux_tensor(r.c, h.G, h.n);
    r.GI = aux_tensor(r.c, h.GI, h.n);
    r.E = h.E;
    r.ids = h.ids;
    r.gi_src = h.gi_src;
    r.last_id = h.last_id;
    r.even = priv_even_mask(r.c, r.E);
}

// The factors -------------------------------------------------------------------------------------

/// n new factors: a row of every exponent, and n columns with a one in the new rows' place.
void aux_new_factors(Host &h, std::size_t count) {
    const std::size_t p = h.ids.size();
    for (std::vector<int> &column : h.E) column.resize(p + count, 0);
    h.last_id = std::max(h.last_id, h.ids.empty() ? int64_t{0}
                                                  : *std::max_element(h.ids.begin(), h.ids.end()));
    for (std::size_t j = 0; j < count; ++j) {
        std::vector<int> column(p + count, 0);
        column[p + j] = 1;
        h.E.push_back(column);
        h.ids.push_back(++h.last_id);
    }
}

/// The independent generators as dependent ones with factors of their own (CORA's remove_GI).
void aux_restructure(Host &h) {
    const std::size_t q = h.GI.size();
    aux_new_factors(h, q);
    h.G.insert(h.G.end(), h.GI.begin(), h.GI.end());
    h.GI.clear();
    h.gi_src.clear();
}

/// The columns of diag(d) with a positive entry as independent generators (all with keep_zero).
void aux_add_diagonal(Host &h, const Column &d, bool keep_zero) {
    for (std::size_t i = 0; i < d.size(); ++i) {
        if (!keep_zero && !(d[i] > 0)) continue;
        Column column(std::size_t(h.n), 0.0);
        column[i] = d[i];
        h.GI.push_back(column);
        h.gi_src.push_back(-1);
    }
}

// The reduction -----------------------------------------------------------------------------------

/// CORA's reducePolyZono: the longest generators (an all-even one counts half) stay, the others
/// become the box d. The shift of an all-even generator g a^e in [0, g] is g / 2 with the radius
/// g / 2 (CORA shifts by g / 4, which misses the top of the generator).
Column aux_reduce(Host &h, double nr_gen) {
    Column d(std::size_t(h.n), 0.0);
    const double N = double(h.n);
    const std::size_t P = h.G.size(), Q = h.GI.size();
    if (!(nr_gen < double(P + Q))) return d;
    const double order = nr_gen / N;
    const std::size_t keep = std::size_t(std::max(0.0, std::floor(N * order - N)));
    if (double(P + Q) > N * order) {
        // the squared length of every generator, an all-even one halved
        const auto even = [&](std::size_t j) {
            return j < P &&
                   std::all_of(h.E[j].begin(), h.E[j].end(), [](int e) { return e % 2 == 0; });
        };
        Column len(P + Q, 0.0);
        for (std::size_t j = 0; j < P + Q; ++j)
            for (const double g : (j < P ? h.G[j] : h.GI[j - P]))
                len[j] += (even(j) ? 0.25 : 1.0) * g * g;
        std::vector<std::size_t> order_of(P + Q);
        std::iota(order_of.begin(), order_of.end(), std::size_t{0});
        std::stable_sort(order_of.begin(), order_of.end(),
                         [&](std::size_t a, std::size_t b) { return len[a] > len[b]; });
        // the removed ones into the shift and the box
        std::vector<bool> removed(P + Q, false);
        for (std::size_t k = std::min(keep, P + Q); k < P + Q; ++k) removed[order_of[k]] = true;
        for (std::size_t j = 0; j < P + Q; ++j) {
            if (!removed[j]) continue;
            const Column &g = j < P ? h.G[j] : h.GI[j - P];
            for (std::size_t i = 0; i < g.size(); ++i) {
                d[i] += even(j) ? 0.5 * std::abs(g[i]) : std::abs(g[i]);
                if (even(j)) h.c[i] += 0.5 * g[i];
            }
        }
        // the kept columns, and the exponents they keep
        std::vector<Column> G, GI;
        Exponents E;
        std::vector<int64_t> src;
        for (std::size_t j = 0; j < P; ++j)
            if (!removed[j]) {
                G.push_back(h.G[j]);
                E.push_back(h.E[j]);
            }
        for (std::size_t j = 0; j < Q; ++j)
            if (!removed[P + j]) {
                GI.push_back(h.GI[j]);
                src.push_back(h.gi_src[j]);
            }
        h.G = G;
        h.GI = GI;
        h.E = E;
        h.gi_src = src;
    }
    // the factors that no generator uses leave
    std::vector<std::size_t> used;
    for (std::size_t r = 0; r < h.ids.size(); ++r)
        if (std::any_of(h.E.begin(), h.E.end(),
                        [&](const std::vector<int> &col) { return col[r] > 0; }))
            used.push_back(r);
    Exponents E;
    for (const std::vector<int> &col : h.E) {
        std::vector<int> kept;
        for (const std::size_t r : used) kept.push_back(col[r]);
        E.push_back(kept);
    }
    std::vector<int64_t> ids;
    for (const std::size_t r : used) ids.push_back(h.ids[r]);
    h.last_id = std::max(h.last_id, h.ids.empty() ? int64_t{0}
                                                  : *std::max_element(h.ids.begin(), h.ids.end()));
    h.E = E;
    h.ids = ids;
    return d;
}

/// The generators of equal exponents added (the first one's place), and those that are zero
/// dropped.
void aux_merge(Host &h) {
    // the first column of each exponent takes the others' generators
    std::map<std::vector<int>, std::size_t> place;
    std::vector<Column> G;
    Exponents E;
    for (std::size_t j = 0; j < h.G.size(); ++j) {
        const auto [it, fresh] = place.emplace(h.E[j], G.size());
        if (fresh) {
            G.push_back(h.G[j]);
            E.push_back(h.E[j]);
            continue;
        }
        for (std::size_t i = 0; i < h.G[j].size(); ++i) G[it->second][i] += h.G[j][i];
    }
    h.G.clear();
    h.E.clear();
    for (std::size_t j = 0; j < G.size(); ++j) {
        if (std::all_of(G[j].begin(), G[j].end(), [](double v) { return v == 0.0; })) continue;
        h.G.push_back(G[j]);
        h.E.push_back(E[j]);
    }
}

/// The independent generators that are zero dropped.
void aux_drop_zero_gi(Host &h) {
    std::vector<Column> GI;
    std::vector<int64_t> src;
    for (std::size_t j = 0; j < h.GI.size(); ++j) {
        if (std::all_of(h.GI[j].begin(), h.GI[j].end(), [](double v) { return v == 0.0; }))
            continue;
        GI.push_back(h.GI[j]);
        src.push_back(h.gi_src[j]);
    }
    h.GI = GI;
    h.gi_src = src;
}

/// CORA's pre-evaluation reduction: down to the number of generators that keeps the powers within
/// max_gens_post, then the independent generators as dependent ones if remove_gi.
void aux_pre_reduce(Host &h, int max_order, bool remove_gi, const NnOptions &options) {
    double limit = double(h.G.size() + h.GI.size());
    if (options.num_generators > 0) limit = std::min(limit, double(options.num_generators));
    const int64_t post =
        options.max_gens_post >= 0 ? options.max_gens_post : 100 * options.num_generators;
    if (options.do_pre_order_reduction && post > 0)
        limit = std::min(limit, std::pow(double(post), 1.0 / double(max_order)));
    if (double(h.G.size() + h.GI.size()) > limit) {
        const Column d = aux_reduce(h, limit);
        aux_add_diagonal(h, d, false);
    }
    if (!h.GI.empty() && remove_gi) aux_restructure(h);
}

} // namespace


// ===========================================  MAIN  =========================================== //

// The layer ---------------------------------------------------------------------------------------

void priv_layer_poly(PrivRows &r, Activation activation, const std::vector<int> &orders,
                     bool errors_in_gi, const NnOptions &options, std::vector<double> *errors) {
    const bool remove_gi = options.remove_gi.value_or(!options.add_approx_error_to_gi);
    const int max_order = *std::max_element(orders.begin(), orders.end());
    // the reduction before the powers, then the range of every neuron
    Host h = aux_load(r);
    aux_pre_reduce(h, max_order, remove_gi, options);
    aux_store(r, h);
    const auto [lower, upper] = priv_bounds(r);
    const std::vector<double> lo = lower.data(), hi = upper.data();
    // the line of the neurons of order 1 (CORALean's), the polynomial of the others
    const bool lines =
        options.poly_method == PolyMethod::singh || options.poly_method == PolyMethod::regression;
    std::vector<double> slope, middle, reach;
    // the lines of all neurons at once
    if (lines) {
        const PrivLine line = priv_line(activation, lower, upper, options);
        const Tensor lo_t = line.offset - line.radius, hi_t = line.offset + line.radius;
        const Tensor mid_t = (lo_t + hi_t) * 0.5;
        slope = line.slope.data();
        middle = mid_t.data();
        reach = (hi_t - mid_t).maximum(mid_t - lo_t).data();
    }
    // the polynomial of every neuron
    std::vector<PrivApprox> approx(orders.size());
    int degree = 1;
    for (std::size_t i = 0; i < orders.size(); ++i) {
        if (lines && orders[i] <= 1)
            approx[i] = {{slope[i], middle[i]}, reach[i]};
        else
            approx[i] = priv_approx_poly(activation, std::min(lo[i], hi[i]), std::max(lo[i], hi[i]),
                                         std::max(orders[i], 1), options.poly_method);
        degree = std::max(degree, int(approx[i].coeffs.size()) - 1);
    }
    // the polynomial of every neuron on its row
    h = aux_load(r);
    // the output generators have the layout of the largest degree
    const int64_t n = h.n, hd = int64_t(h.G.size()), q = int64_t(h.GI.size());
    std::vector<Column> G(std::size_t(priv_power_dep_size(hd, degree)),
                          Column(std::size_t(n), 0.0));
    std::vector<Column> GI(std::size_t(priv_power_ind_size(hd, q, degree)),
                           Column(std::size_t(n), 0.0));
    Column d(std::size_t(n), 0.0);
    // the rows of the neuron i, and its polynomial
    for (int64_t i = 0; i < n; ++i) {
        Column gi_row, g_row;
        for (const Column &col : h.G) g_row.push_back(col[std::size_t(i)]);
        for (const Column &col : h.GI) gi_row.push_back(col[std::size_t(i)]);
        const PrivNeuron y =
            priv_eval_poly(h.c[std::size_t(i)], g_row, gi_row, approx[std::size_t(i)].coeffs);
        h.c[std::size_t(i)] = y.c;
        for (std::size_t k = 0; k < y.G.size(); ++k) G[k][std::size_t(i)] = y.G[k];
        for (std::size_t k = 0; k < y.GI.size(); ++k) GI[k][std::size_t(i)] = y.GI[k];
        d[std::size_t(i)] = approx[std::size_t(i)].d;
    }
    // the exponents of the powers, the generators in their layout
    h.E = priv_power_exponents(h.E, degree);
    h.G = G;
    h.GI = GI;
    h.gi_src.resize(GI.size(), -1);
    // the reduction after the powers, or the exponents merged
    const double nnz = double(std::count_if(d.begin(), d.end(), [](double v) { return v != 0.0; }));
    const double total = double(h.G.size() + h.GI.size()) + nnz;
    if (options.num_generators > 0 && double(options.num_generators) < total) {
        aux_add_diagonal(h, d, true);
        d = aux_reduce(h, double(options.num_generators));
        aux_drop_zero_gi(h);
        if (!h.GI.empty() && remove_gi) aux_restructure(h);
    } else {
        aux_merge(h);
        aux_drop_zero_gi(h);
    }
    // the errors, as independent generators or as new factors
    if (options.use_approx_error) {
        if (errors_in_gi) {
            aux_add_diagonal(h, d, false);
        } else {
            std::size_t count = 0;
            for (const double v : d) count += v > 0 ? 1 : 0;
            aux_new_factors(h, count);
            for (std::size_t i = 0; i < d.size(); ++i) {
                if (!(d[i] > 0)) continue;
                Column column(std::size_t(n), 0.0);
                column[i] = d[i];
                h.G.push_back(column);
            }
        }
    }
    if (errors) *errors = d;
    aux_store(r, h);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
