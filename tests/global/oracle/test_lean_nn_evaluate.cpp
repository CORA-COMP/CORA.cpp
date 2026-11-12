// test_lean_nn_evaluate - NeuralNetwork::evaluate on a zonotope against the CORALean oracle
//
// ORACLE: nn/@neuralNetwork/evaluate.m
// ORACLE: nn/@neuralNetwork/evaluate_.m
// ORACLE: nn/layers/nonlinear/nnReLULayer.m
// ORACLE: nn/layers/nonlinear/nnSigmoidLayer.m
// ORACLE: nn/layers/nonlinear/nnTanhLayer.m
//
// The oracle's nnLayer.evaluateZonotope and neuralNetwork.evaluate at poly_method "singh" (order
// 1): a layer is the exact map by the slope and the box of the offset and the error added as a
// zonotope. Hand-made networks of every activation kind, then the networks of the AINNCS
// instances ACC, TORA (relu, tanh, sigmoid) and Docking on the instances' initial sets, the wide
// ones (TORA relu 100, Docking 256) by their first layers: the oracle takes minutes for all.
// poly_method "regression" and the polynomial zonotope have no float definition (CORALean), so
// computeApproxPoly and evaluatePolyZonotope are not tagged (docs/oracle-requests.md).
//
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/oracle/neural_network.h"
#include "cora/nn/neural_network/neural_network.h"
#include "oracle_testing.h"

#include <cstdlib>
#include <filesystem>
#include <fstream>
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

/// CORA.cpp's and the oracle's output set of the network on Z.
void aux_check_network(const NeuralNetwork &nn, const Zonotope &Z) {
    NnOptions options;
    options.poly_method = PolyMethod::singh;
    const std::vector<lean::NnLayer> layers = aux_lean_layers(nn);
    const lean::Zonotope Y = lean::evaluate_network(layers, lean::Zonotope(Z), "singh");
    oracle::check(nn.evaluate(Z, options),
                  aux_without_bias_generators(Y, layers, Z.generators().shape()[1]));
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
    return test::finish("lean nn evaluate");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
