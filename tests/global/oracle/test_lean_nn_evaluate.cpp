// test_lean_nn_evaluate - NeuralNetwork::evaluate on a zonotope against the CORALean oracle
//
// ORACLE: nn/@neuralNetwork/evaluate.m
// ORACLE: nn/@neuralNetwork/evaluate_.m
// ORACLE: nn/layers/nonlinear/nnReLULayer.m
// ORACLE: nn/layers/nonlinear/nnSigmoidLayer.m
// ORACLE: nn/layers/nonlinear/nnTanhLayer.m
// ORACLE: nn/layers/nonlinear/@nnActivationLayer/evaluatePolyZonotope.m
//
// The oracle's nnLayer.evaluateZonotope and neuralNetwork.evaluate at poly_method "singh" (order
// 1): a layer is the exact map by the slope and the box of the offset and the error added as a
// zonotope. Hand-made networks of every activation kind, then the networks of the AINNCS
// instances ACC, TORA (relu, tanh, sigmoid) and Docking on the instances' initial sets, the wide
// ones (TORA relu 100, Docking 256) by their first layers: the oracle takes minutes for all.
// The same at poly_method "regression" with CORALean's errors, "bracket" (the derivative bracket)
// and "extremum" (the ends and the stationary points): hand-made layers on narrow, wide and
// one-point ranges, and the networks of QUAD, NAV, CartPole and Docking (first layers) on their
// initial sets. The polynomial zonotope in the GI form (add_approx_error_to_GI, a generator per
// neuron also of radius 0) is checked on the same hand-made and instance networks. The extremum
// is also checked on narrow boxes around 0 (the slope is about 1/4) and on boxes whose end lies
// at the stationary point, a few ulps inside and outside.
//
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   13-November-2026
// Reviewed:      ---

#include "cora/global/oracle/neural_network.h"
#include "cora/nn/neural_network/neural_network.h"
#include "oracle_testing.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>

// the folder of the instance data, set before the benchmark header is read
#define CORACPP_BENCHMARK_DATA "."
#include "../../../examples/cpp/benchmarks/benchmark_neur_net.h"

using namespace cora;

namespace oracle = test::oracle;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The layers of a network as the oracle's: a linear layer, then its activation unless identity.
std::vector<lean::NnLayer> aux_lean_layers(const NeuralNetwork &nn) {
    std::vector<lean::NnLayer> layers;
    for (std::size_t k = 0; k < nn.layers().size(); ++k) {
        layers.push_back(lean::NnLayer::linear(nn.layers()[k].W, nn.layers()[k].b));
        switch (nn.activations()[k]) {
        case Activation::relu:
            layers.push_back(lean::NnLayer::relu());
            break;
        case Activation::sigmoid:
            layers.push_back(lean::NnLayer::sigmoid());
            break;
        case Activation::tanh:
            layers.push_back(lean::NnLayer::tanh());
            break;
        case Activation::identity:
            break;
        }
    }
    return layers;
}

/// A random network of the layer sizes with the activation after each layer.
NeuralNetwork aux_random_network(oracle::Random &rnd, const std::vector<int64_t> &sizes,
                                 const std::vector<Activation> &activations) {
    std::vector<NeuralNetwork::Layer> layers;
    for (std::size_t k = 0; k + 1 < sizes.size(); ++k)
        layers.push_back({rnd.matrix(sizes[k + 1], sizes[k]), rnd.vector(sizes[k + 1])});
    return {layers, activations};
}

/// The oracle's set without the zero generators that its linear layers add (CORALean's plus of the
/// bias is the sum with the zonotope of a box, one generator per output, of radius zero).
lean::Zonotope aux_without_bias_generators(const lean::Zonotope &Y,
                                           const std::vector<lean::NnLayer> &layers,
                                           int64_t inputs) {
    const Tensor G = Y.generators().gather();
    // the generators of the input stay, then each activation's; a linear layer's are zero
    std::vector<int64_t> keep;
    keep.reserve(static_cast<size_t>(G.shape()[1]));
    for (int64_t j = 0; j < inputs; ++j) keep.push_back(j);
    int64_t h = inputs, n = G.shape()[0];
    for (const lean::NnLayer &layer : layers) {
        if (layer.kind == "linear")
            n = layer.W.shape()[0];
        else
            for (int64_t i = 0; i < n; ++i) keep.push_back(h + i);
        h += n;
    }
    return lean::Zonotope(Y.center().gather(), G.take(keep, -1));
}

/// The first k layers of the network (the oracle takes minutes for the wide layers).
NeuralNetwork aux_prefix(const NeuralNetwork &nn, std::size_t k) {
    return {std::vector<NeuralNetwork::Layer>(nn.layers().begin(), nn.layers().begin() + k),
            std::vector<Activation>(nn.activations().begin(), nn.activations().begin() + k)};
}

/// The first k layers with the hidden layers cut to their first `width` neurons (the oracle takes
/// minutes for the 256 wide layers of Docking).
NeuralNetwork aux_narrow(const NeuralNetwork &nn, std::size_t k, int64_t width) {
    std::vector<NeuralNetwork::Layer> layers;
    for (std::size_t i = 0; i < k; ++i) {
        const NeuralNetwork::Layer &L = nn.layers()[i];
        const int64_t rows = std::min(width, L.W.shape()[0]),
                      cols = std::min(width, L.W.shape()[1]);
        layers.push_back({L.W.slice(0, 0, rows).slice(1, 0, cols), L.b.slice(0, 0, rows)});
    }
    return {layers,
            std::vector<Activation>(nn.activations().begin(), nn.activations().begin() + k)};
}

/// The options of a method: "" is singh, "bracket" and "extremum" are regression with CORALean's
/// error of that name.
NnOptions aux_options(const std::string &error) {
    NnOptions options;
    options.poly_method = error.empty() ? PolyMethod::singh : PolyMethod::regression;
    options.regression_error =
        error == "extremum" ? RegressionError::extremum : RegressionError::bracket;
    return options;
}

/// CORA.cpp's and the oracle's output set of the network on Z, at the method of the error name.
void aux_check_network(const NeuralNetwork &nn, const Zonotope &Z, const std::string &error = "") {
    const std::vector<lean::NnLayer> layers = aux_lean_layers(nn);
    const lean::Zonotope Y = lean::evaluate_network(layers, lean::Zonotope(Z),
                                                    error.empty() ? "singh" : "regression", error);
    oracle::check(nn.evaluate(Z, aux_options(error)),
                  aux_without_bias_generators(Y, layers, Z.generators().shape()[1]));
}

/// CORA.cpp's and the oracle's polynomial zonotope (c, G, GI) of the network, the errors as
/// independent generators (the oracle's errorForm GI; its linear layers add zero generators).
void aux_check_gi(const std::string &label, const NeuralNetwork &nn, const Tensor &c,
                  const Tensor &G, const Tensor &GI, const std::string &error) {
    NnOptions options = aux_options(error);
    options.add_approx_error_to_gi = true;
    const std::vector<lean::NnLayer> layers = aux_lean_layers(nn);
    const lean::NnPolyResult Y = lean::evaluate_network_poly(
        layers, c, G, GI, error.empty() ? "singh" : "regression", "GI", error);
    // the factors 1..h with the identity exponents
    const int64_t h = G.shape()[1];
    Exponents E(std::size_t(h), std::vector<int>(std::size_t(h), 0));
    std::vector<int64_t> ids;
    for (int64_t j = 0; j < h; ++j) {
        E[std::size_t(j)][std::size_t(j)] = 1;
        ids.push_back(j + 1);
    }
    const PolyZonotope out = nn.evaluate(PolyZonotope(c, G, GI, E, ids), options);
    // the oracle keeps zero generators, CORA.cpp's polynomial zonotope drops them: so do we
    const auto nonzero = [](const Tensor &M) {
        std::vector<int64_t> keep;
        const std::vector<double> v = M.data();
        const int64_t cols = M.shape()[1];
        for (int64_t j = 0; j < cols; ++j)
            for (int64_t i = 0; i < M.shape()[0]; ++i)
                if (v[std::size_t(i * cols + j)] != 0.0) {
                    keep.push_back(j);
                    break;
                }
        return M.take(keep, -1);
    };
    oracle::check(label + " c", out.center(), Y.c);
    oracle::check(label + " G", out.generators(), nonzero(Y.G));
    oracle::check(label + " GI", out.independent_generators(), nonzero(Y.GI));
}

/// CORA.cpp's and the oracle's output set of one layer of the activation on Z.
void aux_check_layer(const std::string &name, Activation activation, const Zonotope &Z,
                     const std::string &error) {
    const int64_t n = Z.center().shape()[0];
    const NeuralNetwork nn({{Tensor::eye(n), Tensor::zeros({n, 1})}}, {activation});
    const lean::NnLayer layer = name == "relu"      ? lean::NnLayer::relu()
                                : name == "sigmoid" ? lean::NnLayer::sigmoid()
                                                    : lean::NnLayer::tanh();
    oracle::check(name + " " + error, nn.evaluate(Z, aux_options(error)),
                  lean::evaluate_layer(layer, lean::Zonotope(Z), "regression", error));
}

/// The upper end u of the box [l, u] at which the stationary point t0 of the regression line
/// reaches u (in double arithmetic, by bisection), 0 where there is none; the tanh's box is the
/// sigmoid's at twice the ends.
double aux_edge(double l, bool tanh) {
    const double scale = tanh ? 2.0 : 1.0;
    const auto sigma = [](double t) { return 1.0 / (1.0 + std::exp(-t)); };
    // t0(u) - u: positive while the stationary point lies beyond the end
    const auto gap = [&](double u) {
        const double a = scale * l, b = scale * u;
        double sum = 0.0;
        for (int i = 0; i < 20; ++i) sum += (2 * i - 19) * sigma(a + i * (b - a) / 19.0);
        const double m = sum / (70.0 * (b - a));
        return 2.0 * std::atanh(std::sqrt(std::max(0.0, 1.0 - 4.0 * m))) - b;
    };
    double lo = 1e-3, hi = 10.0;
    if (!(gap(lo) > 0.0) || !(gap(hi) < 0.0)) return 0.0;
    for (int i = 0; i < 200; ++i) (gap(0.5 * (lo + hi)) > 0.0 ? lo : hi) = 0.5 * (lo + hi);
    return 0.5 * (lo + hi);
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available() || !oracle::serves("neuralNetwork.evaluate")) return 0;
    const std::vector<std::pair<std::string, Activation>> kinds{
        {"relu", Activation::relu}, {"sigmoid", Activation::sigmoid}, {"tanh", Activation::tanh}};

    // one activation layer of every kind on zonotopes with neurons below, above and across 0
    oracle::run("nnLayer.evaluateZonotope", [&](oracle::Random &rnd) {
        for (const auto &[name, activation] : kinds) {
            for (const auto &[n, m] : std::vector<std::pair<int, int>>{{2, 3}, {5, 4}, {12, 12}}) {
                const Zonotope Z(rnd.vector(n) * 2.0, rnd.matrix(n, m));
                const NeuralNetwork nn({{Tensor::eye(n), Tensor::zeros({n, 1})}}, {activation});
                lean::NnLayer layer = name == "relu"      ? lean::NnLayer::relu()
                                      : name == "sigmoid" ? lean::NnLayer::sigmoid()
                                                          : lean::NnLayer::tanh();
                NnOptions options;
                options.poly_method = PolyMethod::singh;
                oracle::check(name, nn.evaluate(Z, options),
                              lean::evaluate_layer(layer, lean::Zonotope(Z), "singh"));
            }
        }
    });

    // networks of several layers, mixed activations, and the last layer without one
    oracle::run("neuralNetwork.evaluate", [&](oracle::Random &rnd) {
        for (const auto &[name, activation] : kinds) {
            const NeuralNetwork nn = aux_random_network(
                rnd, {3, 6, 5, 2}, {activation, activation, Activation::identity});
            aux_check_network(nn, Zonotope(rnd.vector(3), rnd.matrix(3, 4) * 0.3));
        }
        const NeuralNetwork mixed = aux_random_network(
            rnd, {4, 7, 7, 3}, {Activation::relu, Activation::tanh, Activation::sigmoid});
        aux_check_network(mixed, Zonotope(rnd.vector(4), rnd.matrix(4, 9) * 0.2));
    });

    // the networks of the AINNCS instances on their initial sets
    oracle::run("neuralNetwork.evaluate", [&](oracle::Random &) {
        const std::string folder = (std::filesystem::path(__FILE__).parent_path() / "../../.." /
                                    "examples/benchmarks/data")
                                       .lexically_normal()
                                       .string();
        setenv("CORACPP_BENCHMARK_DATA", folder.c_str(), 1);
        using cora::bench::box_zonotope;
        aux_check_network(bench::load_network("acc"),
                          box_zonotope({90, 32, 0, 10, 30, 0}, {110, 32.2, 0, 11, 30.2, 0}));
        const Zonotope tora = box_zonotope({0.6, -0.7, -0.4, 0.5}, {0.7, -0.6, -0.3, 0.6});
        aux_check_network(aux_prefix(bench::load_network("tora_relu"), 1), tora);
        aux_check_network(bench::load_network("tora_tanh"), tora);
        aux_check_network(bench::load_network("tora_sigmoid"), tora);
        // the 256 wide layers of the docking network take the oracle minutes: its first two
        aux_check_network(aux_prefix(bench::load_network("docking"), 2),
                          box_zonotope({9.5, -4.5, 2.1, 1.5}, {9.55, -4.45, 2.11, 1.51}));
    });
    const std::vector<std::string> errors{"bracket", "extremum"};

    // regression with CORALean's errors: one layer on ranges from a point to 60 wide
    oracle::run("nnLayer.evaluateZonotope", [&](oracle::Random &rnd) {
        for (const auto &[name, activation] : kinds) {
            const int n = 8, m = 5;
            const Tensor widths =
                Tensor::from_data({0.0, 1e-9, 1e-3, 0.05, 0.3, 3.0, 30.0, 0.7}, {n, 1});
            const Tensor G = widths.diag().matmul(rnd.matrix(n, m));
            const Zonotope Z(rnd.vector(n) * 2.0, G);
            for (const std::string &error : errors) aux_check_layer(name, activation, Z, error);
        }
    });

    // narrow boxes around 0 (4 m near 1) and the stationary point at an end, a few ulps off
    oracle::run("nnLayer.evaluateZonotope", [&](oracle::Random &) {
        for (const auto &[name, activation] : kinds) {
            if (activation == Activation::relu) continue;
            std::vector<double> lo, hi;
            for (const double w : {1e-300, 1e-100, 1e-20, 1e-12, 1e-9, 1e-6, 1e-3, 0.1, 1.0})
                for (const auto &[a, b] : std::vector<std::pair<double, double>>{
                         {-w, w}, {0.0, w}, {-w, 0.0}, {w, 2 * w}, {-2 * w, -w}, {-w, 3 * w}})
                    lo.push_back(a), hi.push_back(b);
            // the box [l, u] whose stationary point t0 is u, then at its neighbours
            for (const double l : {-0.3, -1.0, -3.0, -8.0, -20.0}) {
                const double u = aux_edge(l, activation == Activation::tanh);
                if (u <= l) continue;
                for (const int ulps : {0, 1, -1, 2, -2, 5, -5, 1000, -1000}) {
                    double v = u;
                    for (int i = 0; i < std::abs(ulps); ++i)
                        v = std::nextafter(v, ulps > 0 ? 1e300 : -1e300);
                    lo.push_back(l), hi.push_back(v);
                    // and the mirror image, where it is -t0 at the lower end
                    lo.push_back(-v), hi.push_back(-l);
                }
            }
            const Zonotope Z = cora::bench::box_zonotope(lo, hi);
            for (const std::string &error : errors) aux_check_layer(name, activation, Z, error);
        }
    });

    // networks of several layers at regression, and in the GI form of a polynomial zonotope
    oracle::run("neuralNetwork.evaluate", [&](oracle::Random &rnd) {
        // a random network of each activation kind with a random zonotope and its GI form
        for (const auto &[name, activation] : kinds) {
            const NeuralNetwork nn = aux_random_network(
                rnd, {3, 6, 5, 2}, {activation, activation, Activation::identity});
            const Zonotope Z(rnd.vector(3), rnd.matrix(3, 4) * 0.3);
            const Tensor G = rnd.matrix(3, 3) * 0.2, GI = rnd.matrix(3, 2) * 0.1;
            aux_check_gi(name + " singh", nn, Z.center(), G, GI, "");
            for (const std::string &error : errors) {
                aux_check_network(nn, Z, error);
                aux_check_gi(name + " " + error, nn, Z.center(), G, GI, error);
            }
        }
        const NeuralNetwork mixed = aux_random_network(
            rnd, {4, 7, 7, 3}, {Activation::relu, Activation::tanh, Activation::sigmoid});
        const Zonotope Z(rnd.vector(4), rnd.matrix(4, 9) * 0.2);
        const Tensor c = rnd.vector(4), G = rnd.matrix(4, 3) * 0.2, GI = rnd.matrix(4, 2) * 0.1;
        for (const std::string &error : errors) {
            aux_check_network(mixed, Z, error);
            aux_check_gi("mixed " + error, mixed, c, G, GI, error);
        }
    });

    // the networks of QUAD, NAV, CartPole and Docking at regression, on the initial sets
    oracle::run("neuralNetwork.evaluate", [&](oracle::Random &) {
        // the folder of the instance data
        const std::string folder = (std::filesystem::path(__FILE__).parent_path() / "../../.." /
                                    "examples/benchmarks/data")
                                       .lexically_normal()
                                       .string();
        setenv("CORACPP_BENCHMARK_DATA", folder.c_str(), 1);
        using cora::bench::box_zonotope;
        std::vector<double> lo(12, 0.0), hi(12, 0.0);
        for (int i = 0; i < 6; ++i) lo[std::size_t(i)] = -0.4, hi[std::size_t(i)] = 0.4;
        const Zonotope quad = box_zonotope(lo, hi);
        const Zonotope nav = box_zonotope({2.9, 2.9, 0, 0}, {3.1, 3.1, 0, 0});
        const Zonotope cart = box_zonotope({-0.04, -0.02, -0.04, -0.02}, {0.04, 0.02, 0.04, 0.02});
        // the first three layers, cut to 32 neurons: the tanh of the third is the first activation
        const Zonotope docking = box_zonotope({70, 70, -0.14, -0.14}, {106, 106, 0.14, 0.14});
        for (const std::string &error : errors) {
            aux_check_network(aux_prefix(bench::load_network("quad"), 2), quad, error);
            aux_check_gi("quad " + error, aux_prefix(bench::load_network("quad"), 2), quad.center(),
                         quad.generators(), Tensor::zeros({12, 1}), error);
            aux_check_network(bench::load_network("nav_point"), nav, error);
            aux_check_gi("nav " + error, bench::load_network("nav_set"), nav.center(),
                         nav.generators(), Tensor::zeros({4, 1}), error);
            aux_check_network(bench::load_network("cartpole"), cart, error);
            aux_check_gi("cartpole " + error, bench::load_network("cartpole"), cart.center(),
                         cart.generators(), Tensor::zeros({4, 1}), error);
            aux_check_network(aux_narrow(bench::load_network("docking"), 3, 32), docking, error);
        }
    });
    return test::finish("lean nn evaluate");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
