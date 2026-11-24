// test_lean_nn_info - the input order and the structure of a dense chain against CORALean
//
// ORACLE: nn/@neuralNetwork/getInputNeuronOrder.m
// ORACLE: nn/@neuralNetwork/getNumNeurons.m
// ORACLE: nn/@neuralNetwork/getNumParams.m
// ORACLE: nn/@neuralNetwork/getRefinableLayers.m
// ORACLE: nn/@neuralNetwork/getOrderPattern.m
// ORACLE: nn/@neuralNetwork/prepareForZonoBatchEval.m
//
// CORA.cpp against the oracle, exactly: input_neuron_order in-order and along the snake of (1, H,
// W) and (C, H, W) images and without a size, and by the sensitivity at random points (the least
// mean |J| first; the oracle ranks by the sum, the same order unless two sums round to one mean,
// which random points do not meet); num_neurons, num_params and refinable_layers of chains with
// identity layers; order_pattern without orders, with one order, one per neuron and another count;
// and prepareForZonoBatchEval's generators, from the batch net of CORA.cpp's verify
// (priv_batch_net, or the ZonoNet at inf input generators): the count and each activation layer's
// error columns.
//
// Authors:       Claude Opus 5.5
// Written:       24-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/oracle/neural_network.h"
#include "cora/nn/neural_network/neural_network.h"
#include "nn/neural_network/private/priv_zono_batch.h"
#include "oracle_testing.h"

#include <cmath>
#include <limits>
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

/// The generators of CORA.cpp's batch net: the count, and per layer the first error generator and
/// their count ((0, 0) without an activation).
lean::ZonoBatchCount aux_count(const NeuralNetwork &nn, double init, double errors) {
    ZonotopeBatchOptions options;
    options.num_approx_err = errors;
    const ZonoNet net = std::isinf(init) ? ZonoNet(nn, init, errors, options.poly_method,
                                                   options.approx_error_order)
                                         : priv_batch_net(nn, int64_t(init), options);
    lean::ZonoBatchCount out{net.q, {}};
    int64_t first = net.q0;
    for (std::size_t k = 0; k < nn.layers().size(); ++k) {
        if (nn.activations()[k] == Activation::identity) {
            out.errors.emplace_back(0, 0);
            continue;
        }
        out.errors.emplace_back(first, net.slots[k]);
        first += net.slots[k];
    }
    return out;
}

/// The orders as the oracle's integers.
std::vector<std::vector<int64_t>> aux_wide(const std::vector<std::vector<int>> &orders) {
    std::vector<std::vector<int64_t>> out;
    out.reserve(orders.size());
    for (const std::vector<int> &o : orders) out.emplace_back(o.begin(), o.end());
    return out;
}

/// The patterns of CORA.cpp and of the oracle: the same layers, each the same orders.
void aux_check_pattern(const std::string &what, const std::vector<std::vector<int>> &mine,
                       const std::vector<std::vector<int64_t>> &theirs) {
    const std::vector<std::vector<int64_t>> wide = aux_wide(mine);
    oracle::check(what + " layers", int64_t(wide.size()), int64_t(theirs.size()));
    for (std::size_t k = 0; k < wide.size() && k < theirs.size(); ++k)
        oracle::check(what + " layer " + std::to_string(k), wide[k], theirs[k]);
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    const Activation id = Activation::identity, relu = Activation::relu;
    const double inf = std::numeric_limits<double>::infinity();

    // The input order -----------------------------------------------------------------------------

    // the pixels in order, along the snake of (C, H, W) images, then by the sensitivity
    oracle::run("neuralNetwork.getInputNeuronOrder", [&](oracle::Random &rnd) {
        const std::vector<std::vector<int64_t>> sizes{{1, 3, 4}, {2, 3, 4}, {3, 4, 2}};
        for (const std::vector<int64_t> &size : sizes) {
            const NeuralNetwork nn = aux_chain(rnd, {size[0] * size[1] * size[2], 3}, {relu});
            const Tensor x = rnd.point(nn.input_dim());
            for (const std::string method : {"in-order", "snake"})
                oracle::check(method + " " + std::to_string(nn.input_dim()),
                              nn.input_neuron_order(method, x, size),
                              lean::input_neuron_order(nn, method, x, size));
        }
        const NeuralNetwork plain = aux_chain(rnd, {5, 3}, {relu});
        oracle::check("in-order no size", plain.input_neuron_order("in-order", Tensor(), {}),
                      lean::input_neuron_order(plain, "in-order", Tensor(), {}));
        // the sensitivity
        const std::vector<std::pair<std::vector<int64_t>, std::vector<int64_t>>> cases{
            {{6, 4, 3}, {}}, {{12, 5, 3}, {1, 3, 4}}, {{12, 4, 2}, {3, 2, 2}}, {{4, 3}, {}}};
        for (std::size_t t = 0; t < cases.size(); ++t) {
            const std::vector<int64_t> &chain = cases[t].first;
            const NeuralNetwork nn = chain.size() == 3 ? aux_chain(rnd, chain, {relu, id})
                                                       : aux_chain(rnd, chain, {relu});
            for (int p = 0; p < 3; ++p) {
                const Tensor x = rnd.point(nn.input_dim());
                oracle::check("sensitivity " + std::to_string(t) + " point " + std::to_string(p),
                              nn.input_neuron_order("sensitivity", x, cases[t].second),
                              lean::input_neuron_order(nn, "sensitivity", x, cases[t].second));
            }
        }
    });

    // The structure -------------------------------------------------------------------------------

    // the counts and the refinable layers of chains with identity layers between and after
    const std::vector<std::pair<std::vector<int64_t>, std::vector<Activation>>> chains{
        {{3, 2}, {relu}},
        {{4, 5, 3}, {relu, id}},
        {{3, 4, 4, 2}, {id, relu, id}},
        {{2, 3, 2, 3, 2}, {relu, relu, id, Activation::sigmoid}}};
    for (const std::string op : {"getNumNeurons", "getNumParams", "getRefinableLayers"})
        oracle::run("neuralNetwork." + op, [&](oracle::Random &rnd) {
            for (std::size_t t = 0; t < chains.size(); ++t) {
                const NeuralNetwork nn = aux_chain(rnd, chains[t].first, chains[t].second);
                const std::string what = op + " " + std::to_string(t);
                if (op == "getNumNeurons")
                    oracle::check(what, nn.num_neurons(), lean::num_neurons(nn));
                else if (op == "getNumParams")
                    oracle::check(what, nn.num_params(), lean::num_params(nn));
                else
                    oracle::check(what, nn.refinable_layers(), lean::refinable_layers(nn));
            }
        });

    // no orders, then per layer none, one, one per neuron and another count (their maximum)
    oracle::run("neuralNetwork.getOrderPattern", [&](oracle::Random &rnd) {
        const NeuralNetwork nn = aux_chain(rnd, {3, 4, 3, 3, 2}, {relu, id, relu, relu});
        aux_check_pattern("no orders", nn.order_pattern(), lean::order_pattern(nn, {}));
        const std::vector<std::vector<std::vector<int>>> cases{
            {{2}, {}, {1, 3, 2}, {4, 2}}, {{}, {5}, {3}, {1, 2}}, {{1, 2, 3, 2}, {7}}};
        for (std::size_t t = 0; t < cases.size(); ++t)
            aux_check_pattern("orders " + std::to_string(t), nn.order_pattern(cases[t]),
                              lean::order_pattern(nn, aux_wide(cases[t])));
    });

    // the input's generators (inf, fewer, as many, more) and the errors' (inf, fewer, none)
    oracle::run("neuralNetwork.prepareForZonoBatchEval", [&](oracle::Random &rnd) {
        const NeuralNetwork nn = aux_chain(rnd, {4, 5, 3, 3, 2}, {relu, id, relu, id});
        for (const double init : {inf, 2.0, 4.0, 7.0})
            for (const double errors : {inf, 2.0, 0.0}) {
                const lean::ZonoBatchCount mine = aux_count(nn, init, errors);
                const lean::ZonoBatchCount theirs = lean::prepare_zono_batch_eval(nn, init, errors);
                // the count, then each layer's first error generator and their count
                const std::string what =
                    "init " + std::to_string(init) + " errors " + std::to_string(errors);
                oracle::check(what + " numGen", mine.num_gen, theirs.num_gen);
                oracle::check(what + " layers", int64_t(mine.errors.size()),
                              int64_t(theirs.errors.size()));
                for (std::size_t k = 0; k < mine.errors.size() && k < theirs.errors.size(); ++k) {
                    oracle::check(what + " first " + std::to_string(k), mine.errors[k].first,
                                  theirs.errors[k].first);
                    oracle::check(what + " count " + std::to_string(k), mine.errors[k].second,
                                  theirs.errors[k].second);
                }
            }
    });
    return test::finish("lean nn info");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
