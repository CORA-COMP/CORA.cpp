// test_lean_nn_order_k - the layers of order k against the CORALean oracle, bit for bit
//
// ORACLE: nn/layers/nonlinear/@nnActivationLayer/evaluatePolyZonotopeNeuron.m
// ORACLE: nn/layers/nonlinear/@nnActivationLayer/evaluatePolyZonotope.m
//
// The powers of one neuron (priv_eval_poly, priv_power_exponents) against the oracle's
// polyZonotope.evalPowers on random rows of orders 1 to 5, with and without independent
// generators and with repeated exponents; then the layer of order k (priv_layer_poly, the errors
// as independent generators, no generator limit) against the oracle's layer "poly", given the fits
// CORA.cpp computes (CORA's fits are data to CORALean, so is their error d): random sets through
// sigmoid, tanh and ReLU at every poly_method, the neurons of one layer at different orders. The
// oracle keeps the generators that are zero (the padded powers, the merged columns that cancel,
// the errors of radius 0), CORA.cpp drops them: they are dropped from its answer before the
// comparison, every other column is compared in its order.
// Only the oracle's zero columns go (their exponents with them): a zero column of CORA.cpp fails.
//
// Authors:       Claude Opus 5.5
// Written:       16-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/oracle/neural_network.h"
#include "cora/global/oracle/poly_zonotope.h"
#include "nn/neural_network/private/priv_activation_poly.h"
#include "nn/neural_network/private/priv_layers.h"
#include "nn/neural_network/private/priv_poly_power.h"
#include "oracle_testing.h"

#include <algorithm>

using namespace cora;

namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The shape of a random set: n dimensions, h dependent generators over p factors, q independent.
struct Dims {
    int64_t n, h, p, q;
};

/// A random set with exponents 0..2 (none constant), every third column repeating an earlier one.
PolyZonotope aux_set(oracle::Random &rnd, const Dims &s) {
    Exponents E;
    for (int64_t l = 0; l < s.h; ++l) {
        std::vector<int> col(std::size_t(s.p), 0);
        for (int &e : col) e = static_cast<int>(rnd.uniform(0, 3));
        if (std::all_of(col.begin(), col.end(), [](int e) { return e == 0; })) col[0] = 1;
        E.push_back(l % 3 == 2 ? E[std::size_t(l - 2)] : col);
    }
    std::vector<int64_t> id(std::size_t(s.p));
    for (int64_t i = 0; i < s.p; ++i) id[std::size_t(i)] = i + 1;
    // small generators, so that the powers stay in range
    return {rnd.vector(s.n),   rnd.matrix(s.n, s.h) * 0.5, rnd.matrix(s.n, s.q) * 0.25, E, id,
            detail::Verbatim{}};
}

/// Row i of an (n, m) tensor.
std::vector<double> aux_row(const Tensor &T, int64_t i) {
    const int64_t m = T.shape()[1];
    const std::vector<double> data = T.data();
    return {data.begin() + i * m, data.begin() + (i + 1) * m};
}

/// The (n, m) tensor of n rows of m entries.
Tensor aux_tensor(const std::vector<std::vector<double>> &rows, int64_t m) {
    std::vector<double> data;
    for (const std::vector<double> &row : rows) data.insert(data.end(), row.begin(), row.end());
    return Tensor::from_data(data, {int64_t(rows.size()), m});
}

/// The coefficients (n, k + 1) lowest degree first, of lists highest first padded to degree k.
Tensor aux_lowest_first(const std::vector<std::vector<double>> &highest_first, int k) {
    std::vector<std::vector<double>> rows;
    for (const std::vector<double> &a : highest_first) {
        std::vector<double> row(a.rbegin(), a.rend());
        row.resize(std::size_t(k) + 1, 0.0);
        rows.push_back(row);
    }
    return aux_tensor(rows, k + 1);
}

/// Whether column j of an (n, m) tensor is zero.
bool aux_zero_column(const std::vector<double> &data, int64_t n, int64_t m, int64_t j) {
    for (int64_t i = 0; i < n; ++i)
        if (data[std::size_t(i * m + j)] != 0.0) return false;
    return true;
}

/// The columns of an (n, m) tensor that are kept.
Tensor aux_columns(const Tensor &T, const std::vector<int64_t> &kept) {
    const int64_t n = T.shape()[0], m = T.shape()[1];
    const std::vector<double> data = T.data();
    std::vector<double> out;
    for (int64_t i = 0; i < n; ++i)
        for (const int64_t j : kept) out.push_back(data[std::size_t(i * m + j)]);
    return Tensor::from_data(out, {n, int64_t(kept.size())});
}

/// The oracle's set without its zero generators, dependent (with their exponents) and independent.
PolyZonotope aux_nonzero(const PolyZonotope &P) {
    const int64_t n = P.dim(), h = P.generators().shape()[1];
    const int64_t q = P.independent_generators().shape()[1];
    const std::vector<double> g = P.generators().data(), gi = P.independent_generators().data();
    // the dependent columns that are not zero, then the independent ones
    std::vector<int64_t> keep_g, keep_gi;
    Exponents E;
    for (int64_t j = 0; j < h; ++j)
        if (!aux_zero_column(g, n, h, j)) {
            keep_g.push_back(j);
            E.push_back(P.exponents()[std::size_t(j)]);
        }
    for (int64_t j = 0; j < q; ++j)
        if (!aux_zero_column(gi, n, q, j)) keep_gi.push_back(j);
    return {P.center(),
            aux_columns(P.generators(), keep_g),
            aux_columns(P.independent_generators(), keep_gi),
            E,
            P.ids(),
            detail::Verbatim{}};
}

/// CORA.cpp's powers of every row of the set, as one set: priv_eval_poly row by row.
PolyZonotope aux_powers(const PolyZonotope &P, const std::vector<std::vector<double>> &coeffs) {
    const int64_t n = P.dim();
    const int k = int(coeffs[0].size()) - 1;
    const int64_t h = P.generators().shape()[1], q = P.independent_generators().shape()[1];
    const int64_t dep = priv_power_dep_size(h, k), ind = priv_power_ind_size(h, q, k);
    // each row through its polynomial
    std::vector<double> c;
    std::vector<std::vector<double>> G, GI;
    for (int64_t i = 0; i < n; ++i) {
        const PrivNeuron y =
            priv_eval_poly(P.center().data()[std::size_t(i)], aux_row(P.generators(), i),
                           aux_row(P.independent_generators(), i), coeffs[std::size_t(i)]);
        c.push_back(y.c);
        G.push_back(y.G);
        GI.push_back(y.GI);
    }
    return {Tensor::from_data(c, {n, 1}),
            aux_tensor(G, dep),
            aux_tensor(GI, ind),
            priv_power_exponents(P.exponents(), k),
            P.ids(),
            detail::Verbatim{}};
}

/// Random coefficients of degree k, highest first, one list per row.
std::vector<std::vector<double>> aux_coeffs(oracle::Random &rnd, int64_t n, int k) {
    std::vector<std::vector<double>> out(static_cast<std::size_t>(n));
    for (std::vector<double> &a : out)
        for (int j = 0; j <= k; ++j) a.push_back(rnd.uniform(-1.0, 1.0));
    return out;
}

/// The tracked rows of a set, as evaluate.cpp's layers of order k take them.
PrivRows aux_rows(const PolyZonotope &X) {
    PrivRows r;
    r.c = X.center();
    r.G = X.generators();
    r.GI = X.independent_generators();
    r.even = priv_even_mask(r.c, X.exponents());
    r.track = true;
    r.E = X.exponents();
    r.ids = X.ids();
    r.last_id = *std::max_element(r.ids.begin(), r.ids.end());
    for (int64_t j = 0; j < r.GI.shape()[1]; ++j) r.gi_src.push_back(j);
    return r;
}

/// The layer of order k on the set in CORA.cpp and in the oracle, given CORA.cpp's fits.
void aux_check_layer(const PolyZonotope &X, Activation activation, const std::vector<int> &orders,
                     PolyMethod method, const std::string &label) {
    const NnOptions options{.poly_method = method};
    // the fits CORA.cpp makes: on the ranges of the rows, before the layer changes them
    PrivRows r = aux_rows(X);
    const auto [lower, upper] = priv_bounds(r);
    const std::vector<double> lo = lower.data(), hi = upper.data();
    std::vector<std::vector<double>> coeffs;
    int degree = 1;
    for (std::size_t i = 0; i < orders.size(); ++i) {
        coeffs.push_back(priv_approx_poly(activation, std::min(lo[i], hi[i]),
                                          std::max(lo[i], hi[i]), orders[i], method)
                             .coeffs);
        degree = std::max(degree, int(coeffs.back().size()) - 1);
    }
    std::vector<double> d;
    priv_layer_poly(r, activation, orders, true, options, &d);
    const PolyZonotope mine(r.c, r.G, r.GI, r.E, r.ids, detail::Verbatim{});
    // the oracle's layer on the same fits and errors
    const lean::NnLayer layer = lean::NnLayer::poly(aux_lowest_first(coeffs, degree),
                                                    Tensor::from_data(d, {int64_t(d.size()), 1}));
    const PolyZonotope theirs = lean::evaluate_layer(layer, lean::PolyZonotope(X)).gather();
    oracle::check(label, mine, lean::PolyZonotope(aux_nonzero(theirs)));
}

/// The shapes the ops are tried on.
const std::vector<Dims> kDims = {{1, 1, 1, 0}, {2, 2, 2, 1}, {2, 4, 2, 0}, {3, 3, 3, 2}};

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;

    // the powers of every row, orders 1 to 5
    oracle::run("polyZonotope.evalPowers", [](oracle::Random &rnd) {
        for (const Dims &s : kDims)
            for (int k = 1; k <= 5; ++k) {
                const PolyZonotope P = aux_set(rnd, s);
                const std::vector<std::vector<double>> a = aux_coeffs(rnd, s.n, k);
                oracle::check("order " + std::to_string(k), aux_powers(P, a),
                              lean::PolyZonotope(P).eval_powers(aux_lowest_first(a, k)));
            }
    });

    // the layer at every method and activation, one order for all neurons and mixed orders
    oracle::run("nnLayer.evaluatePolyZonotope", [](oracle::Random &rnd) {
        const std::vector<PolyMethod> methods = {PolyMethod::regression,
                                                 PolyMethod::ridge_regression, PolyMethod::taylor,
                                                 PolyMethod::bernstein, PolyMethod::throw_catch};
        for (const Dims &s : kDims)
            for (const Activation a : {Activation::sigmoid, Activation::tanh, Activation::relu})
                for (const PolyMethod m : methods) {
                    // CORA has no Taylor series and no throw-catch fit of the ReLU
                    if (a == Activation::relu &&
                        (m == PolyMethod::taylor || m == PolyMethod::throw_catch))
                        continue;
                    const PolyZonotope X = aux_set(rnd, s);
                    std::vector<int> mixed(std::size_t(s.n));
                    for (int64_t i = 0; i < s.n; ++i) mixed[std::size_t(i)] = 2 + int(i % 3);
                    aux_check_layer(X, a, std::vector<int>(std::size_t(s.n), 3), m, "order 3");
                    aux_check_layer(X, a, mixed, m, "orders 2 to 4");
                }
    });
    return test::finish("lean nn order k");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
