// test_lean_nn_forms - the network forms and the passes of a ReLU chain against CORALean
//
// ORACLE: nn/layers/nonlinear/@nnActivationLayer/getMergeBuckets.m
// ORACLE: nn/layers/nonlinear/nnGroupSortLayer.m
// ORACLE: nn/layers/linear/nnGeneratorReductionLayer.m
// ORACLE: nn/@neuralNetwork/getNormalForm.m
// ORACLE: nn/@neuralNetwork/propagateBounds.m
// ORACLE: nn/@neuralNetwork/backprop.m
// ORACLE: nn/@neuralNetwork/backpropIntervalBatch.m
// ORACLE: nn/@neuralNetwork/calcSensitivity.m
//
// CORA.cpp's port against the oracle, bit for bit: merge_buckets of the ReLU, sigmoid and tanh
// (nnLayer.mergeBuckets); a network of group_sort_layer at group sizes 1 to 8 on points with
// ties (nnLayer.groupSort; no entry is +-0); reduce_generators of one zonotope to no, some and all
// of its generators, with ties of the lengths (nnLayer.generatorReduction); normal_form of dense
// chains with identity layers between and after the activations (neuralNetwork.normalForm);
// propagate_bounds from a ReLU layer to the next one, without stored bounds there, met with them,
// and from a layer without a ReLU after it; backprop at a point, backprop_interval_batch over a box
// and sensitivity of ReLU and identity chains. A zero of a sum of zero products has the sign of
// its order, so -0 and 0 are one value in the passes (x + 0).
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

// The networks ------------------------------------------------------------------------------------

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

/// A point result against both bounds of the oracle's box, -0 as 0.
void aux_check_box(const std::string &what, const Tensor &mine, const lean::Interval &theirs) {
    const auto [lo, hi] = theirs.gather();
    oracle::check(what + " inf", aux_zero(mine), aux_zero(lo));
    oracle::check(what + " sup", aux_zero(mine), aux_zero(hi));
}

/// A box [c - |r|, c + |r|] of n values.
NnBounds aux_box(oracle::Random &rnd, int64_t n) {
    const Tensor c = rnd.point(n), r = rnd.point(n).abs() * 0.5;
    return {c - r, c + r};
}

/// The bounds of propagate_bounds in CORA.cpp and in the oracle: the same ones set.
void aux_check_bounds(const std::string &what, const NeuralNetwork &nn,
                      const std::vector<NnBounds> &bounds, int64_t i) {
    const std::vector<NnBounds> mine = nn.propagate_bounds(bounds, i);
    const std::vector<NnBounds> theirs = lean::propagate_bounds(nn, bounds, i);
    oracle::check(what + " count", int64_t(mine.size()), int64_t(theirs.size()));
    for (std::size_t k = 0; k < mine.size() && k < theirs.size(); ++k) {
        const std::string at = what + " layer " + std::to_string(k);
        oracle::check(at + " set", mine[k].l.defined(), theirs[k].l.defined());
        if (!mine[k].l.defined() || !theirs[k].l.defined()) continue;
        oracle::check(at + " l", mine[k].l, theirs[k].l);
        oracle::check(at + " u", mine[k].u, theirs[k].u);
    }
}

/// The pass `op` of the network ("backprop" at a point, "backpropIntervalBatch" over a box,
/// "sensitivity" at a point) in CORA.cpp and in the oracle.
void aux_check_pass(const std::string &op, const std::string &what, oracle::Random &rnd,
                    const NeuralNetwork &nn) {
    const int64_t n = nn.input_dim(), m = nn.output_dim();
    const Tensor x = rnd.point(n), g = rnd.point(m);
    if (op == "backprop") {
        aux_check_box(what, nn.backprop(x.expand_dims(0), g.expand_dims(0)).squeeze(0),
                      lean::backprop(nn, x, g));
    } else if (op == "sensitivity") {
        aux_check_box(what, nn.sensitivity(x), lean::sensitivity(nn, x));
    } else {
        // the bounds of the gradient over a box, from bounds at the output
        const NnBounds box = aux_box(rnd, n), grad = aux_box(rnd, m);
        const auto [lo, hi] =
            nn.backprop_interval_batch(box.l.expand_dims(0), box.u.expand_dims(0),
                                       grad.l.expand_dims(0), grad.u.expand_dims(0));
        const auto [glo, ghi] = lean::backprop_interval(nn, box.l, box.u, grad.l, grad.u);
        aux_check_box(what + " gl", lo.squeeze(0), glo);
        aux_check_box(what + " gu", hi.squeeze(0), ghi);
    }
}

/// The pass on ReLU and identity chains of one to three layers, narrow ones (the oracle's passes
/// take seconds at three layers of a few neurons).
void aux_check_chains(const std::string &op, oracle::Random &rnd) {
    const Activation id = Activation::identity, relu = Activation::relu;
    aux_check_pass(op, op + " one layer", rnd, aux_chain(rnd, {3, 2}, {relu}));
    aux_check_pass(op, op + " relu id", rnd, aux_chain(rnd, {4, 5, 2}, {relu, id}));
    aux_check_pass(op, op + " id relu", rnd, aux_chain(rnd, {3, 4, 3}, {id, relu}));
    aux_check_pass(op, op + " relu relu id", rnd, aux_chain(rnd, {2, 3, 2, 2}, {relu, relu, id}));
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    if (!oracle::available()) return 0;

    // The forms -----------------------------------------------------------------------------------

    oracle::run("nnLayer.mergeBuckets", [&](oracle::Random &) {
        const std::vector<std::pair<Activation, std::string>> kinds{
            {Activation::relu, "relu"},
            {Activation::sigmoid, "sigmoid"},
            {Activation::tanh, "tanh"}};
        for (const auto &[a, kind] : kinds) {
            const std::vector<double> mine = NeuralNetwork::merge_buckets(a);
            oracle::check("mergeBuckets " + kind, Tensor::from_data(mine, {int64_t(mine.size())}),
                          lean::merge_buckets(kind));
        }
    });

    // the points of 7 values with two ties, sorted in groups (the last one shorter or padded)
    oracle::run("nnLayer.groupSort", [&](oracle::Random &rnd) {
        std::vector<double> data = rnd.point(7).data();
        data[4] = data[1];
        data[6] = data[5];
        const Tensor x = Tensor::from_data(data, {7});
        for (const int64_t g : {1, 2, 3, 4, 7, 8}) {
            const NeuralNetwork nn(NeuralNetwork::group_sort_layer(7, g));
            oracle::check("groupSort " + std::to_string(g),
                          nn.evaluate(x.expand_dims(0)).squeeze(0), lean::group_sort(x, g));
        }
    });

    // a zonotope of 3 dimensions and 8 generators, two of the same length as others
    oracle::run("nnLayer.generatorReduction", [&](oracle::Random &rnd) {
        std::vector<double> data = rnd.matrix(8, 3).data();
        for (std::size_t i = 0; i < 3; ++i) {
            data[6 * 3 + i] = -data[1 * 3 + i];
            data[7 * 3 + 2 - i] = data[3 * 3 + i];
        }
        const Tensor c = rnd.point(3), G = Tensor::from_data(data, {8, 3});
        const ZonotopeBatch Z{c.expand_dims(0), G.expand_dims(0), Tensor()};
        for (const int64_t max_gens : {3, 4, 6, 9, 11, 12}) {
            const ZonotopeBatch Y = NeuralNetwork::reduce_generators(Z, max_gens);
            oracle::check("generatorReduction " + std::to_string(max_gens),
                          Zonotope(Y.c.squeeze(0), Y.G.squeeze(0)),
                          lean::generator_reduction(lean::Zonotope(Zonotope(c, G)), max_gens));
        }
    });

    // dense chains: linear layers fused before an activation, between two, and after the last
    oracle::run("neuralNetwork.normalForm", [&](oracle::Random &rnd) {
        const Activation id = Activation::identity, relu = Activation::relu;
        const std::vector<std::pair<std::vector<int64_t>, std::vector<Activation>>> chains{
            {{3, 4, 5, 2}, {relu, relu, id}},
            {{3, 4, 4, 5, 2}, {id, relu, id, id}},
            {{2, 3, 4, 3, 3, 2}, {id, Activation::sigmoid, id, Activation::tanh, id}},
            {{4, 3, 3}, {id, id}}};
        for (std::size_t t = 0; t < chains.size(); ++t) {
            const NeuralNetwork nn = aux_chain(rnd, chains[t].first, chains[t].second);
            const NeuralNetwork mine = nn.normal_form();
            const std::vector<lean::NnDense> theirs = lean::normal_form(nn);
            const std::string what = "normalForm " + std::to_string(t);
            oracle::check(what + " layers", mine.num_layers(), int64_t(theirs.size()));
            // each fused layer and the name of its activation
            for (std::size_t k = 0; k < theirs.size() && int64_t(k) < mine.num_layers(); ++k) {
                const std::string at = what + " layer " + std::to_string(k);
                oracle::check(at + " W", mine.layers()[k].W, theirs[k].W);
                oracle::check(at + " b", mine.layers()[k].b, theirs[k].b);
                const Activation a = mine.activations()[k];
                const std::string name = a == relu                  ? "relu"
                                         : a == Activation::sigmoid ? "sigmoid"
                                         : a == Activation::tanh    ? "tanh"
                                                                    : "identity";
                oracle::check(at + " activation", name == theirs[k].activation, true);
            }
        }
    });

    // The passes ----------------------------------------------------------------------------------

    // relu, id, relu, id, relu: the bounds of layer 0 reach layer 2, those of 2 reach layer 4
    oracle::run("neuralNetwork.propagateBounds", [&](oracle::Random &rnd) {
        const Activation id = Activation::identity, relu = Activation::relu;
        const NeuralNetwork nn = aux_chain(rnd, {3, 4, 5, 4, 3, 2}, {relu, id, relu, id, relu});
        std::vector<NnBounds> bounds(5);
        bounds[0] = aux_box(rnd, 4);
        aux_check_bounds("propagateBounds from 0", nn, bounds, 0);
        // stored bounds at layer 2 that cut the propagated ones on some neurons
        const std::vector<NnBounds> after = nn.propagate_bounds(bounds, 0);
        const Tensor mid = (after[2].l + after[2].u) * 0.5;
        const Tensor half = Tensor::from_data({1.0, 0.0, 1.0, 0.0}, {4});
        bounds[2] = {after[2].l + (mid - after[2].l).multiply(half),
                     after[2].u - (after[2].u - mid).multiply(half.ones_like() - half)};
        aux_check_bounds("propagateBounds met", nn, bounds, 0);
        aux_check_bounds("propagateBounds from 2", nn, bounds, 2);
        // an identity layer and the last ReLU change nothing
        bounds[4] = aux_box(rnd, 2);
        aux_check_bounds("propagateBounds from 1", nn, bounds, 1);
        aux_check_bounds("propagateBounds from 4", nn, bounds, 4);
    });

    // the gradients and the Jacobian of ReLU and identity chains
    for (const std::string op : {"backprop", "backpropIntervalBatch", "sensitivity"})
        oracle::run("neuralNetwork." + op, [&](oracle::Random &rnd) { aux_check_chains(op, rnd); });
    return test::finish("lean nn forms");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
