// test_lean_nn_weights - the weights and the orders of a dense chain against CORALean
//
// ORACLE: nn/@neuralNetwork/castWeights.m
// ORACLE: nn/@neuralNetwork/generateRandom.m
// ORACLE: nn/@neuralNetwork/initWeights.m
// ORACLE: nn/@neuralNetwork/resetApproxOrder.m
// ORACLE: nn/@neuralNetwork/refine.m
//
// CORA.cpp against the oracle, bit for bit: the network in single precision (to(float32)) as the
// oracle's rounding to nearest in ieee:binary32; random and re-initialized (glorot) networks on
// the units of CORA.cpp's Rng, which the test draws again from the same seed in the same order (a
// uniform at [0, 1) is the unit itself), shi on the standard normals of the same stream, which
// the oracle only scales (Box-Muller's log, cos and sin stay on the platform's libm, on both
// sides); order_pattern without orders as resetApproxOrder; and one step of refine of every type
// and heuristic kind on random heuristics.
//
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   25-November-2026
// Reviewed:      ---

#include "cora/global/oracle/neural_network.h"
#include "cora/global/rng.h"
#include "cora/nn/neural_network/neural_network.h"
#include "oracle_testing.h"

#include <algorithm>
#include <array>
#include <optional>
#include <string>
#include <utility>
#include <vector>

using namespace cora;

namespace oracle = test::oracle;
using Layer = NeuralNetwork::Layer;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A dense chain of the sizes (in, then each layer's outputs) and the activations, one per layer.
NeuralNetwork aux_chain(oracle::Random &rnd, const std::vector<int64_t> &sizes,
                        const std::vector<Activation> &activations) {
    std::vector<Layer> layers;
    for (std::size_t k = 0; k + 1 < sizes.size(); ++k)
        layers.push_back(Layer::linear(rnd.matrix(sizes[k + 1], sizes[k]) * 0.5,
                                       rnd.point(sizes[k + 1]) * 0.25));
    return NeuralNetwork(layers, activations);
}

/// The next n units of the stream, one draw of the Rng (Rng::uniform at [0, 1)).
Tensor aux_units(Rng &rng, int64_t n) {
    std::vector<double> u(static_cast<std::size_t>(n));
    rng.uniform(u.data(), u.size(), 0.0, 1.0);
    return Tensor::from_data(u, {n});
}

/// The next n standard normals of the stream, one draw of the Rng (Rng::normal at scale 1).
Tensor aux_normals(Rng &rng, int64_t n) {
    std::vector<double> z(static_cast<std::size_t>(n));
    rng.normal(z.data(), z.size(), 1.0);
    return Tensor::from_data(z, {n});
}

/// The orders as the oracle's integers.
std::vector<std::vector<int64_t>> aux_wide(const std::vector<std::vector<int>> &orders) {
    std::vector<std::vector<int64_t>> out;
    out.reserve(orders.size());
    for (const std::vector<int> &o : orders) out.emplace_back(o.begin(), o.end());
    return out;
}

/// The orders of CORA.cpp against the oracle's, layer by layer.
void aux_check_orders(const std::string &what, const std::vector<std::vector<int>> &mine,
                      const std::vector<std::vector<int64_t>> &theirs) {
    const std::vector<std::vector<int64_t>> wide = aux_wide(mine);
    oracle::check(what + " layers", int64_t(wide.size()), int64_t(theirs.size()));
    for (std::size_t k = 0; k < wide.size() && k < theirs.size(); ++k)
        oracle::check(what + " layer " + std::to_string(k), wide[k], theirs[k]);
}

/// A box that is a point, as CORA.cpp's tensor of that shape.
void aux_check_point(const std::string &what, const Tensor &mine, const lean::Interval &theirs) {
    const auto [lo, hi] = theirs.gather();
    oracle::check(what + " inf", mine.reshape(lo.shape()), lo);
    oracle::check(what + " sup", mine.reshape(hi.shape()), hi);
}

// The random networks -----------------------------------------------------------------------------

/// generateRandom: CORA.cpp's network of the options against the oracle's on the same units.
void aux_check_random(const RandomOptions &o) {
    const NeuralNetwork nn = NeuralNetwork::random(o);
    // the units of the sizes drawn, in CORA's order, then of every layer's W and b
    Rng rng(o.seed);
    const std::array<int64_t, 4> sizes{o.inputs, o.outputs, o.layers, o.hidden};
    std::vector<double> size_draws;
    for (const int64_t s : sizes)
        if (s == 0) size_draws.push_back(aux_units(rng, 1).data()[0]);
    // one draw of W's units and one of b's per layer
    std::vector<std::pair<Tensor, Tensor>> draws;
    for (const Layer &L : nn.layers()) {
        const int64_t m = L.outputs(), n = L.inputs();
        const Tensor W = aux_units(rng, m * n).reshape({m, n});
        draws.emplace_back(W, aux_units(rng, m));
    }
    const lean::NnRandom theirs = lean::generate_random(sizes, size_draws, draws);
    const std::string what = "generateRandom seed " + std::to_string(o.seed);
    test::check(theirs.sizes[0] == nn.input_dim() && theirs.sizes[1] == nn.output_dim() &&
                    theirs.sizes[2] == nn.num_layers(),
                what + ": the sizes");
    for (std::size_t k = 0; k < nn.layers().size(); ++k) {
        aux_check_point(what + " W" + std::to_string(k), nn.layers()[k].W, theirs.layers[k].first);
        aux_check_point(what + " b" + std::to_string(k), nn.layers()[k].b, theirs.layers[k].second);
    }
}

/// initWeights: CORA.cpp's weights of the method against the oracle's on the same units.
void aux_check_init(const NeuralNetwork &nn, const std::string &method, uint64_t seed,
                    const std::vector<int64_t> &which) {
    const NeuralNetwork fresh = nn.init_weights(method, seed, which);
    // one draw per layer drawn again: m n units (glorot) or m n standard normals (shi)
    const bool shi = method == "shi";
    Rng rng(seed);
    std::vector<Tensor> draws;
    for (std::size_t k = 0; k < nn.layers().size(); ++k) {
        if (!which.empty() && std::find(which.begin(), which.end(), int64_t(k)) == which.end())
            continue;
        const int64_t q = nn.layers()[k].W.numel();
        draws.push_back(shi ? aux_normals(rng, q) : aux_units(rng, q));
    }
    const std::vector<std::optional<lean::Interval>> theirs =
        lean::init_weights(nn, method, which, draws, shi);
    for (std::size_t k = 0; k < nn.layers().size(); ++k) {
        if (!theirs[k]) continue;
        aux_check_point("initWeights " + method + " W" + std::to_string(k), fresh.layers()[k].W,
                        *theirs[k]);
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    const Activation id = Activation::identity, relu = Activation::relu;
    const std::vector<std::pair<std::vector<int64_t>, std::vector<Activation>>> chains{
        {{3, 2}, {relu}}, {{4, 5, 3}, {relu, id}}, {{2, 3, 4, 2}, {relu, relu, id}}};

    // the weights in single precision
    oracle::run("neuralNetwork.castWeights", [&](oracle::Random &rnd) {
        // the cast on a backend with float32 (the lean backend computes in float64 only)
        const BackendScope host("eigen");
        for (std::size_t t = 0; t < chains.size(); ++t) {
            const NeuralNetwork nn = aux_chain(rnd, chains[t].first, chains[t].second);
            const NeuralNetwork single = nn.to(DType::float32);
            const auto theirs = lean::cast_weights(nn, "ieee:binary32");
            for (std::size_t k = 0; k < nn.layers().size(); ++k) {
                const std::string what =
                    "cast chain " + std::to_string(t) + " layer " + std::to_string(k);
                oracle::check(what + " W", single.layers()[k].W.to(DType::float64),
                              theirs[k].first);
                oracle::check(what + " b", single.layers()[k].b.to(DType::float64).expand_dims(-1),
                              theirs[k].second);
            }
        }
    });
    // random networks, the sizes given and drawn
    oracle::run("neuralNetwork.generateRandom", [&](oracle::Random &) {
        aux_check_random({.inputs = 3, .outputs = 2, .layers = 2, .hidden = 4, .seed = 3});
        for (const uint64_t seed : {1u, 2u, 5u, 11u}) aux_check_random({.seed = seed});
        aux_check_random({.inputs = 2, .outputs = 0, .layers = 3, .hidden = 0, .seed = 17});
    });
    // the weights drawn again (sqrt and arithmetic only on the draws), all layers and the last
    oracle::run("neuralNetwork.initWeights", [&](oracle::Random &rnd) {
        for (std::size_t t = 0; t < chains.size(); ++t) {
            const NeuralNetwork nn = aux_chain(rnd, chains[t].first, chains[t].second);
            for (const char *method : {"glorot", "shi"}) {
                aux_check_init(nn, method, 1 + t, {});
                aux_check_init(nn, method, 7 + t, {int64_t(nn.layers().size()) - 1});
            }
        }
    });
    // the orders: every one 1, then one step of refine
    oracle::run("neuralNetwork.resetApproxOrder", [&](oracle::Random &rnd) {
        for (const auto &[sizes, acts] : chains) {
            const NeuralNetwork nn = aux_chain(rnd, sizes, acts);
            aux_check_orders("resetApproxOrder", nn.order_pattern(), lean::reset_approx_order(nn));
        }
    });
    oracle::run("neuralNetwork.refine", [&](oracle::Random &rnd) {
        const std::vector<std::pair<RefineType, std::string>> types{
            {RefineType::all, "all"}, {RefineType::layer, "layer"}, {RefineType::neuron, "neuron"}};
        const std::vector<std::pair<RefineHeuristic, std::string>> kinds{
            {RefineHeuristic::approx_error, "approx_error"},
            {RefineHeuristic::layer_bias, "layer_bias"},
            {RefineHeuristic::all, "all"}};
        for (const auto &[sizes, acts] : chains) {
            const NeuralNetwork nn = aux_chain(rnd, sizes, acts);
            // a heuristic per refinable neuron, and orders below and at the maximum
            std::vector<std::vector<double>> heu(nn.layers().size());
            std::vector<std::vector<int>> orders(nn.layers().size());
            for (std::size_t k = 0; k < heu.size(); ++k) {
                if (acts[k] == id) continue;
                heu[k] = rnd.point(nn.layers()[k].outputs()).abs().data();
                for (std::size_t j = 0; j < heu[k].size(); ++j) orders[k].push_back(1 + int(j % 3));
            }
            for (const auto &[type, tname] : types)
                for (const auto &[kind, kname] : kinds)
                    for (const int max_order : {2, 5}) {
                        const RefineOptions o{max_order, type, kind, 0.25};
                        const std::string what = "refine " + tname + " " + kname;
                        aux_check_orders(
                            what, nn.refine(orders, heu, o),
                            lean::refine(nn, aux_wide(orders), heu, tname, kname, max_order, 0.25));
                    }
        }
    });
    return test::finish("lean nn weights");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
