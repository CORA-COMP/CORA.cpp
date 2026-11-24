// test_lean_nn_fgsm - the fast gradient sign method of a ReLU chain against CORALean
//
// ORACLE: nn/@neuralNetwork/computeFGSMAttack.m
//
// CORA.cpp's compute_fgsm_attack without a mask against the oracle
// (neuralNetwork.computeFGSMAttack), bit for bit: ReLU and identity chains of one to three layers,
// random points, one-hot targets and a few radii; the gradient is backprop of softmax(y) - t, the
// step epsilon along its sign (0 where the gradient is 0). One batch row per request.
//
// Authors:       Claude Opus 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/oracle/neural_network.h"
#include "cora/nn/neural_network/neural_network.h"
#include "oracle_testing.h"

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

/// The one-hot target of class i among m.
Tensor aux_onehot(int64_t i, int64_t m) {
    std::vector<double> t(static_cast<std::size_t>(m), 0.0);
    t[static_cast<std::size_t>(i)] = 1.0;
    return Tensor::from_data(t, {m});
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    const Activation id = Activation::identity, relu = Activation::relu;

    // each chain at two points, every class the target once, three radii
    oracle::run("neuralNetwork.computeFGSMAttack", [&](oracle::Random &rnd) {
        const std::vector<std::pair<std::vector<int64_t>, std::vector<Activation>>> chains{
            {{3, 2}, {relu}},
            {{4, 5, 3}, {relu, id}},
            {{3, 4, 3}, {id, relu}},
            {{2, 3, 2, 3}, {relu, relu, id}}};
        for (std::size_t t = 0; t < chains.size(); ++t) {
            const NeuralNetwork nn = aux_chain(rnd, chains[t].first, chains[t].second);
            const int64_t n = nn.input_dim(), m = nn.output_dim();
            // a random point per target class
            for (int64_t i = 0; i < m; ++i) {
                const Tensor x = rnd.point(n), target = aux_onehot(i, m);
                for (const double eps : {0.01, 0.1, 0.375}) {
                    const std::string what = "fgsm chain " + std::to_string(t) + " class " +
                                             std::to_string(i) + " eps " + std::to_string(eps);
                    const Tensor mine =
                        nn.compute_fgsm_attack(x.expand_dims(0), target.expand_dims(0), eps)
                            .squeeze(0);
                    const lean::Interval theirs = lean::compute_fgsm_attack(nn, x, target, eps);
                    const auto [lo, hi] = theirs.gather();
                    oracle::check(what + " inf", mine.reshape(lo.shape()), lo);
                    oracle::check(what + " sup", mine.reshape(hi.shape()), hi);
                }
            }
        }
    });
    return test::finish("lean nn fgsm");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
