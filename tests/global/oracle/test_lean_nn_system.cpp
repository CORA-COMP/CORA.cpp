// test_lean_nn_system - a ReLU chain as a system, and its explanation, against CORALean
//
// ORACLE: nn/@neuralNetwork/nonlinearSysDT.m
// ORACLE: nn/@neuralNetwork/explain.m
//
// CORA.cpp against the oracle, bit for bit: NetworkSystemDT's output with the offsets u (W h + b)
// + u_k, C (the sensitivity at x) and every column block of D = dy / du_k; and explain's method
// standard on the order given, in-order and by the sensitivity, the class of the point as label,
// a few radii and an image size (1, 2, 2). A zero of a sum of zero products may differ in its
// sign, so the system's values are compared as x + 0.
//
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
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

/// A point result against a point box of the oracle, both as x + 0 (the sign of a zero).
void aux_check_point(const std::string &what, const Tensor &mine, const lean::Interval &theirs) {
    const auto [lo, hi] = theirs.gather();
    oracle::check(what + " inf", mine.reshape(lo.shape()) + 0.0, lo + 0.0);
    oracle::check(what + " sup", mine.reshape(hi.shape()) + 0.0, hi + 0.0);
}

/// The class of the point: the largest output.
int64_t aux_class(const NeuralNetwork &nn, const Tensor &x) {
    const std::vector<double> y = nn.evaluate(x.expand_dims(0)).data();
    int64_t best = 0;
    for (std::size_t i = 1; i < y.size(); ++i)
        if (y[i] > y[std::size_t(best)]) best = int64_t(i);
    return best;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    const Activation id = Activation::identity, relu = Activation::relu;
    const std::vector<std::pair<std::vector<int64_t>, std::vector<Activation>>> chains{
        {{3, 2}, {relu}},
        {{4, 5, 3}, {relu, id}},
        {{3, 4, 3}, {id, relu}},
        {{2, 3, 2, 3}, {relu, relu, id}}};

    // the system's output and Jacobians at random points and offsets
    oracle::run("neuralNetwork.nonlinearSysDT", [&](oracle::Random &rnd) {
        for (std::size_t t = 0; t < chains.size(); ++t) {
            const NeuralNetwork nn = aux_chain(rnd, chains[t].first, chains[t].second);
            const NetworkSystemDT sys = nn.nonlinear_sys_dt();
            for (int rep = 0; rep < 3; ++rep) {
                const Tensor x = rnd.point(sys.num_states());
                const Tensor u = rnd.point(sys.num_inputs()) * (rep == 0 ? 0.0 : 0.5);
                const std::string what =
                    "nonlinearSysDT chain " + std::to_string(t) + " rep " + std::to_string(rep);
                const lean::NnSystem theirs = lean::nonlinear_sys_dt(nn, x, u);
                aux_check_point(what + " y", sys.output(x, u), theirs.y);
                aux_check_point(what + " C", sys.output_jacobian_x(x), theirs.C);
                // D, one column block per layer
                const Tensor D = sys.output_jacobian_u(x, u);
                int64_t first = 0;
                for (std::size_t k = 0; k < theirs.D.size(); ++k) {
                    const int64_t m = nn.layers()[k].outputs();
                    aux_check_point(what + " D" + std::to_string(k), D.slice(1, first, first + m),
                                    theirs.D[k]);
                    first += m;
                }
                oracle::check(what + " blocks", first, sys.num_inputs());
            }
        }
    });
    // the freed features of the method standard
    oracle::run("neuralNetwork.explain", [&](oracle::Random &rnd) {
        for (std::size_t t = 0; t < chains.size(); ++t) {
            const NeuralNetwork nn = aux_chain(rnd, chains[t].first, chains[t].second);
            const int64_t n = nn.input_dim();
            for (const double eps : {0.1, 0.75, 2.0}) {
                const Tensor x = rnd.point(n);
                const int64_t label = aux_class(nn, x);
                for (const std::string method : {"in-order", "sensitivity"}) {
                    ExplainOptions o;
                    o.feat_order_method = method;
                    const Explanation mine = nn.explain(x, label, eps, o);
                    const std::string what = "explain chain " + std::to_string(t) + " " + method +
                                             " eps " + std::to_string(eps);
                    oracle::check(what, mine.freed,
                                  lean::explain(nn, x, label, eps, mine.feat_order));
                }
            }
        }
        // an image of one channel, 2 x 2 pixels
        const NeuralNetwork img = aux_chain(rnd, {4, 3, 2}, {relu, id});
        const Tensor x = rnd.point(4);
        ExplainOptions o;
        o.feat_order_method = "snake";
        o.input_size = {1, 2, 2};
        const Explanation mine = img.explain(x, aux_class(img, x), 0.2, o);
        oracle::check("explain image", mine.freed,
                      lean::explain(img, x, aux_class(img, x), 0.2, mine.feat_order, {1, 2, 2}));
    });
    return test::finish("lean nn system");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
