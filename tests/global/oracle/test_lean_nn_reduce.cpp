// test_lean_nn_reduce - the reduced network and the constrained zonotope of a ReLU chain
//
// ORACLE: nn/@neuralNetwork/computeReducedNetwork.m
// ORACLE: nn/layers/nonlinear/@nnActivationLayer/evaluateConZonotope.m
//
// CORA.cpp against the oracle, bit for bit: compute_reduced_network with static buckets (ReLU
// chains, the last layer the identity) at the default tolerance, a wide one and target rates: the
// reduced layers, their error intervals, the merged neurons, the rates and the output set; and
// evaluate_con_zonotope of a zonotope without constraints at the box bounds: c, the generators,
// A and b, or the empty set. The set of a reduced layer goes through an identity map first in
// CORA.cpp, so a zero may differ in its sign: values are compared as x + 0.
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

/// A dense chain of the sizes (in, then each layer's outputs) and the activations, one per layer;
/// the biases shifted by `shift` (negative: more neurons that stay at 0).
NeuralNetwork aux_chain(oracle::Random &rnd, const std::vector<int64_t> &sizes,
                        const std::vector<Activation> &activations, double shift = 0.0) {
    std::vector<Layer> layers;
    for (std::size_t k = 0; k + 1 < sizes.size(); ++k)
        layers.push_back(Layer::linear(rnd.matrix(sizes[k + 1], sizes[k]) * 0.5,
                                       rnd.point(sizes[k + 1]) * 0.25 + shift));
    return NeuralNetwork(layers, activations);
}

/// Two tensors of the same entries, as x + 0 (the sign of a zero).
void aux_check(const std::string &what, const Tensor &mine, const Tensor &theirs) {
    oracle::check(what, mine.reshape(theirs.shape()) + 0.0, theirs + 0.0);
}

/// CORA.cpp's reduction against the oracle's, field by field.
void aux_check_reduced(const std::string &what, const NeuralNetwork &nn, const Zonotope &X,
                       const ReduceOptions &o) {
    const ReducedNetwork mine = nn.compute_reduced_network(X, o);
    const lean::NnReduced theirs =
        lean::compute_reduced_network(nn, X, o.bucket_tol, o.reduction_rate, o.min_searches);
    oracle::check(what + " layers", mine.net.num_layers(), int64_t(theirs.layers.size()));
    for (std::size_t k = 0; k < theirs.layers.size(); ++k) {
        const std::string at = what + " layer " + std::to_string(k);
        aux_check(at + " W", mine.net.layers()[k].W, theirs.layers[k].first);
        aux_check(at + " b", mine.net.layers()[k].b, theirs.layers[k].second);
        // the error interval, where the layer has one
        test::check(mine.offset_center[k].defined() == theirs.d[k].has_value(),
                    at + ": an error interval in both or neither");
        if (theirs.d[k]) {
            aux_check(at + " dc", mine.offset_center[k], theirs.d[k]->first);
            aux_check(at + " dr", mine.offset_radius[k], theirs.d[k]->second);
        }
        // the merged neurons, 1 for merged
        if (k + 1 < theirs.layers.size() && mine.merged[k].defined()) {
            std::vector<int64_t> mask;
            for (const double v : mine.merged[k].data()) mask.push_back(v != 0.0 ? 1 : 0);
            oracle::check(at + " merged", mask, theirs.merged[k]);
        }
    }
    aux_check(what + " rate",
              Tensor::from_data(mine.reduction_rate, {int64_t(mine.reduction_rate.size())}),
              theirs.rate);
    const auto [c, G] = std::pair(theirs.Z.center().gather(), theirs.Z.generators().gather());
    aux_check(what + " c", mine.output.center(), c);
    aux_check(what + " G", mine.output.generators().matrix_transpose(), G);
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;
    const Activation id = Activation::identity, relu = Activation::relu;
    const std::vector<std::pair<std::vector<int64_t>, std::vector<Activation>>> chains{
        {{3, 6, 2}, {relu, id}}, {{2, 8, 5, 2}, {relu, relu, id}}, {{4, 10, 3}, {relu, id}}};

    // the default tolerance, a wide one, and target rates
    oracle::run("neuralNetwork.computeReducedNetwork", [&](oracle::Random &rnd) {
        for (std::size_t t = 0; t < chains.size(); ++t) {
            const NeuralNetwork nn = aux_chain(rnd, chains[t].first, chains[t].second, -0.25);
            const int64_t n = nn.input_dim();
            const Zonotope X(rnd.point(n) * 0.5, rnd.matrix(2, n) * 0.1);
            const std::string what = "reduce chain " + std::to_string(t);
            aux_check_reduced(what + " default", nn, X, {});
            // a tolerance that merges more, then the rates
            ReduceOptions wide;
            wide.bucket_tol = {0.25};
            aux_check_reduced(what + " wide", nn, X, wide);
            for (const double rate : {0.5, 0.75, 1.0}) {
                ReduceOptions o;
                o.reduction_rate = {rate};
                aux_check_reduced(what + " rate " + std::to_string(rate), nn, X, o);
            }
        }
    });
    // the network on a zonotope as a constrained zonotope
    oracle::run("neuralNetwork.evaluateConZonotope", [&](oracle::Random &rnd) {
        for (std::size_t t = 0; t < chains.size(); ++t) {
            const NeuralNetwork nn = aux_chain(rnd, chains[t].first, chains[t].second);
            const int64_t n = nn.input_dim();
            for (const double r : {0.01, 0.2, 1.0}) {
                const Tensor c = rnd.point(n) * 0.5, G = rnd.matrix(n, n) * r;
                const ConZonotope Y = nn.evaluate_con_zonotope(ConZonotope(c, G));
                const lean::NnConZono theirs =
                    lean::evaluate_con_zonotope(nn, c, G.matrix_transpose());
                const std::string what =
                    "conZonotope chain " + std::to_string(t) + " r " + std::to_string(r);
                const bool empty = Y.nr_generators() == 0 && Y.A().size(-2) == 1;
                test::check(empty == theirs.empty, what + ": empty in both or neither");
                // the center, the generators and the constraints, where there are any
                if (theirs.empty) continue;
                aux_check(what + " c", Y.c(), theirs.c);
                aux_check(what + " G", Y.generators().matrix_transpose(), theirs.G);
                if (theirs.A.numel() > 0) {
                    aux_check(what + " A", Y.A(), theirs.A);
                    aux_check(what + " b", Y.b(), theirs.b);
                }
            }
        }
    });
    return test::finish("lean nn reduce");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
