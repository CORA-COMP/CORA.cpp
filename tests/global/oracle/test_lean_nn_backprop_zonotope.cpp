// test_lean_nn_backprop_zonotope - the zonotope batch's gradients with the lines held, CORALean
//
// ORACLE: nn/@neuralNetwork/backpropZonotopeBatch.m
// ORACLE: nn/@neuralNetwork/backpropZonotopeBatch_.m
// ORACLE: nn/layers/nonlinear/@nnActivationLayer/backpropZonotopeBatch.m
// ORACLE: nn/layers/linear/nnLinearLayer.m
//
// CORA.cpp's backprop_zonotope_batch (exact false, every error a generator) against the oracle,
// bit for bit: ReLU and identity chains of one to three layers, a zonotope of h generators
// forward, random gradients of the output's center and of all its generators backward; the
// gradients at the input's center and generators and the slopes of every ReLU layer's lines
// (priv_verify_line at the recorded bounds). The ops of one layer (nnLayer.backpropZonotopeBatch):
// a linear layer as a network of that layer alone, a ReLU as one behind the identity map. CORA.cpp
// keeps a batch in rows (B, n) and (B, q, n); one row per request, the oracle's columns transposed.
// A zero of a sum of zero products has the sign of its order, so -0 and 0 are one value (x + 0).
//
// Authors:       Claude Opus 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/oracle/neural_network.h"
#include "cora/nn/neural_network/neural_network.h"
#include "nn/neural_network/private/priv_zono_batch.h"
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

/// The value with -0 as 0.
Tensor aux_zero(const Tensor &x) { return x + 0.0; }

/// A point result against both bounds of the oracle's box (in its shape), -0 as 0.
void aux_check_box(const std::string &what, const Tensor &mine, const lean::Interval &theirs) {
    const auto [lo, hi] = theirs.gather();
    const Tensor m = aux_zero(mine.reshape(lo.shape()));
    oracle::check(what + " inf", m, aux_zero(lo));
    oracle::check(what + " sup", m, aux_zero(hi));
}

/// The slopes of the ReLU layers' lines on the zonotope (c, G) (n, (h, n)), undefined for another
/// layer (priv_verify_line at the recorded bounds), and the output's generator count.
std::pair<std::vector<Tensor>, int64_t> aux_slopes(const NeuralNetwork &nn, const Tensor &c,
                                                   const Tensor &G) {
    const ZonoNet net = priv_batch_net(nn, G.shape()[0], ZonotopeBatchOptions{});
    const ZonoPropagation out =
        net.forward(c.expand_dims(0), Tensor(), G.expand_dims(0), Tensor(), nullptr, true);
    std::vector<Tensor> slopes(nn.layers().size());
    std::size_t next = 0;
    for (std::size_t k = 0; k < slopes.size(); ++k) {
        if (nn.activations()[k] != Activation::relu) continue;
        const ZonoLayerRecord &rec = out.layers[next++];
        slopes[k] = priv_verify_line(Activation::relu, rec.l, rec.u, net.poly_method).slope;
    }
    return {slopes, out.G.shape()[1]};
}

/// The slopes of CORA.cpp and of the oracle: the same layers set, the same values.
void aux_check_slopes(const std::string &what, const std::vector<Tensor> &mine,
                      const std::vector<Tensor> &theirs) {
    oracle::check(what + " slopes", int64_t(theirs.size()), int64_t(mine.size()));
    for (std::size_t k = 0; k < mine.size() && k < theirs.size(); ++k) {
        const std::string at = what + " slope " + std::to_string(k);
        oracle::check(at + " set", theirs[k].defined(), mine[k].defined());
        if (!mine[k].defined() || !theirs[k].defined()) continue;
        const Tensor &lean_s = theirs[k];
        oracle::check(at, aux_zero(mine[k].reshape(lean_s.shape())), aux_zero(lean_s));
    }
}

/// The gradients of nn at the zonotope (c, G) (n, (h, n)) from random ones at the output, and the
/// slopes of the ReLU layers, in CORA.cpp and in the oracle.
void aux_check_net(const std::string &what, oracle::Random &rnd, const NeuralNetwork &nn,
                   const Tensor &c, const Tensor &G) {
    const auto [slopes, q] = aux_slopes(nn, c, G);
    const Tensor g_c = rnd.point(nn.output_dim()), g_gen = rnd.matrix(q, nn.output_dim());
    const ZonotopeBatch mine = nn.backprop_zonotope_batch(c.expand_dims(0), G.expand_dims(0),
                                                          g_c.expand_dims(0), g_gen.expand_dims(0));
    const lean::NnZonoGrad theirs =
        lean::backprop_zonotope_batch(nn, c, G.matrix_transpose(), g_c, g_gen.matrix_transpose());
    aux_check_box(what + " g_c", mine.c.squeeze(0), theirs.g_c);
    aux_check_box(what + " g_gen", mine.G.squeeze(0).matrix_transpose(), theirs.g_gen);
    aux_check_slopes(what, slopes, theirs.slopes);
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    const Activation id = Activation::identity, relu = Activation::relu;

    // The networks --------------------------------------------------------------------------------

    // ReLU and identity chains on zonotopes of fewer, as many and more generators than inputs
    oracle::run("neuralNetwork.backpropZonotopeBatch", [&](oracle::Random &rnd) {
        const auto zonotope = [&](int64_t n, int64_t h) {
            return std::pair{rnd.point(n), rnd.matrix(h, n) * 0.3};
        };
        const std::vector<std::pair<std::vector<int64_t>, std::vector<Activation>>> chains{
            {{3, 2}, {relu}},
            {{3, 4, 2}, {relu, id}},
            {{2, 3, 3}, {id, relu}},
            {{2, 3, 2, 2}, {relu, relu, id}}};
        for (std::size_t t = 0; t < chains.size(); ++t) {
            const NeuralNetwork nn = aux_chain(rnd, chains[t].first, chains[t].second);
            const int64_t n = nn.input_dim();
            for (const int64_t h : {int64_t(1), n, n + 2}) {
                const auto [c, G] = zonotope(n, h);
                aux_check_net("chain " + std::to_string(t) + " h " + std::to_string(h), rnd, nn, c,
                              G);
            }
        }
    });

    // The layers ----------------------------------------------------------------------------------

    // a linear layer alone (W' g_c and W' g_gen at any set), then a ReLU behind the identity map
    oracle::run("nnLayer.backpropZonotopeBatch", [&](oracle::Random &rnd) {
        const Tensor W = rnd.matrix(3, 4), b = rnd.point(3);
        const NeuralNetwork linear({Layer::linear(W, b)}, {id});
        for (const int64_t q : {1, 3, 5}) {
            const Tensor c = rnd.point(4), G = rnd.matrix(q, 4), g_c = rnd.point(3);
            const Tensor g_gen = rnd.matrix(q, 3);
            const ZonotopeBatch mine = linear.backprop_zonotope_batch(
                c.expand_dims(0), G.expand_dims(0), g_c.expand_dims(0), g_gen.expand_dims(0));
            const lean::NnZonoGrad theirs = lean::backprop_zonotope_batch_layer(
                lean::NnLayer::linear(W, b), Tensor(), Tensor(), g_c, g_gen.matrix_transpose());
            const std::string what = "linear q " + std::to_string(q);
            aux_check_box(what + " g_c", mine.c.squeeze(0), theirs.g_c);
            aux_check_box(what + " g_gen", mine.G.squeeze(0).matrix_transpose(), theirs.g_gen);
        }
        // the ReLU's input set is the identity's output
        const int64_t n = 4;
        const NeuralNetwork nn({Layer::linear(Tensor::eye(n), Tensor::zeros({n}))}, {relu});
        for (const int64_t h : {2, 4, 6}) {
            const Tensor c = rnd.point(n), G = rnd.matrix(h, n) * 0.4;
            const Tensor g_c = rnd.point(n), g_gen = rnd.matrix(h + n, n);
            const ZonotopeBatch mine = nn.backprop_zonotope_batch(
                c.expand_dims(0), G.expand_dims(0), g_c.expand_dims(0), g_gen.expand_dims(0));
            const lean::NnZonoGrad theirs = lean::backprop_zonotope_batch_layer(
                lean::NnLayer::relu(), c, G.matrix_transpose(), g_c, g_gen.matrix_transpose());
            const std::string what = "relu h " + std::to_string(h);
            aux_check_box(what + " g_c", mine.c.squeeze(0), theirs.g_c);
            aux_check_box(what + " g_gen", mine.G.squeeze(0).matrix_transpose(), theirs.g_gen);
            aux_check_slopes(what, aux_slopes(nn, c, G).first, theirs.slopes);
        }
    });
    return test::finish("lean nn backprop zonotope");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
