// test_neural_network_verify_witness - CORA's verify with and without the three soundness repairs
//
// 60 random 2-input ReLU networks (2-k-k-2, k in 4..8), each with an unsafe set y_0 - y_1 <= t at
// the 0.2% quantile of sampled outputs: every instance has unsafe inputs, so "unsat" is wrong. The
// options are CORA's preset with one neuron split, eight ReLU rows and three error slots (an
// interval center), which reach all three unsound steps of CORA's verify. With the repairs on
// (the default), no verdict is wrong; with drop_dummy_duplicates off (CORA's isDummySplit: the
// children of a dummy split counted as verified) some are, so the test sees what it guards. The
// counts of each repair off are printed (the Python reference found 2 for the stale factors and 40
// for the dummy splits of its 60 networks, 0 for the interval center).
//
// CORA: nn/@neuralNetwork/verify.m
// CORA authors: Lukas Koller
// Authors:       Claude Opus 5.5
// Written:       21-November-2026
// Last update:   22-November-2026
// Reviewed:      ---

#include "cora/global/rng.h"
#include "cora/nn/neural_network/neural_network.h"
#include "cora/nn/neural_network/verify.h"
#include "testing.h"

#include <algorithm>
#include <cmath>
#include <iostream>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr int kNetworks = 60;
constexpr int64_t kSamples = 50000;

/// The instance of a seed: the network and the property with its sampled unsafe inputs.
struct Instance {
    NeuralNetwork net;
    NnProperty property;
};

Instance aux_instance(int seed) {
    Rng rng(uint64_t(1000 + seed));
    std::vector<double> k(2);
    rng.uniform(k.data(), 2, 4.0, 9.0);
    const std::vector<int64_t> sizes{2, int64_t(k[0]), int64_t(k[1]), 2};
    std::vector<NeuralNetwork::Layer> layers;
    for (std::size_t i = 0; i < 3; ++i) {
        std::vector<double> W(static_cast<std::size_t>(sizes[i + 1] * sizes[i])),
            b(static_cast<std::size_t>(sizes[i + 1]));
        rng.normal(W.data(), W.size(), 1.0 / std::sqrt(double(sizes[i])));
        rng.normal(b.data(), b.size(), 0.5);
        // float32 weights, as an ONNX network has
        for (double &w : W) w = double(float(w));
        for (double &v : b) v = double(float(v));
        layers.push_back(
            {Tensor::from_data(W, {sizes[i + 1], sizes[i]}), Tensor::from_data(b, {sizes[i + 1]})});
    }
    const NeuralNetwork net(layers, {Activation::relu, Activation::relu, Activation::identity});
    // the 0.2% quantile of y_0 - y_1 over the box [-1, 1]^2
    std::vector<double> x(static_cast<std::size_t>(2 * kSamples), 0.0);
    rng.uniform(x.data(), x.size(), -1.0, 1.0);
    const std::vector<double> y = net.evaluate(Tensor::from_data(x, {kSamples, 2})).data();
    std::vector<double> diff(static_cast<std::size_t>(kSamples), 0.0);
    for (std::size_t i = 0; i < diff.size(); ++i) diff[i] = y[2 * i] - y[2 * i + 1];
    std::nth_element(diff.begin(), diff.begin() + kSamples / 500, diff.end());
    NnProperty p;
    p.lo = Tensor::from_data(std::vector<double>{-1.0, -1.0}, {1, 2});
    p.hi = Tensor::from_data(std::vector<double>{1.0, 1.0}, {1, 2});
    p.A = Tensor::from_data(std::vector<double>{1.0, -1.0}, {1, 1, 2});
    p.b = Tensor::from_data(std::vector<double>{diff[std::size_t(kSamples / 500)]}, {1, 1});
    p.poly_case = {0};
    return {net, p};
}

/// The options of the witness search: CORA's preset with splits, ReLU rows and an interval center.
NnVerifyOptions aux_options() {
    NnVerifyOptions o = cora_vnncomp_options("witness");
    o.falsification_method = "center";
    o.falsification_order = "bound-first";
    o.bab_falsification_method = "center";
    o.num_neuron_splits = 1;
    o.num_relu_constraints = 8;
    o.num_approx_err = 3;
    o.max_verif_iter = 200;
    o.mini_batch_size = 4;
    return o;
}

/// The number of wrong unsat verdicts of the 60 instances under the options.
int aux_wrong(const NnVerifyOptions &o) {
    int wrong = 0;
    for (int seed = 0; seed < kNetworks; ++seed) {
        const Instance inst = aux_instance(seed);
        if (inst.net.verify(inst.property, 20, o).status == "unsat") ++wrong;
    }
    return wrong;
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    const NnVerifyOptions repaired = aux_options();
    const int wrong = aux_wrong(repaired);
    std::cout << "repaired: " << wrong << " wrong of " << kNetworks << "\n";
    check(wrong == 0, "the repaired verify answers no instance with unsafe inputs unsat");
    // each repair off, CORA's step
    NnVerifyOptions dummy = repaired, stale = repaired, center = repaired;
    dummy.drop_dummy_duplicates = false;
    stale.project_stale_factors = false;
    center.remember_point_splits_only = false;
    const int wrong_dummy = aux_wrong(dummy);
    std::cout << "dummy splits as CORA: " << wrong_dummy << " wrong\n"
              << "stale factors as CORA: " << aux_wrong(stale) << " wrong\n"
              << "interval-center splits as CORA: " << aux_wrong(center) << " wrong\n";
    check(wrong_dummy > 0, "CORA's dummy splits answer some instance with unsafe inputs unsat");
    // Phase 2's slopes, backward bound, dual certificate, look-ahead and chord rows: none unsat
    NnVerifyOptions phase2 = repaired;
    phase2.bounding_method = "zonotope-alpha";
    phase2.backward_bound = true;
    phase2.dual_certificate = true;
    phase2.neuron_split_heuristic = "fsb";
    phase2.relu_constraint_faces = "triangle";
    const int wrong2 = aux_wrong(phase2);
    phase2.alpha_method = "crown";
    phase2.optimize_gap = 0.5;
    const int wrong3 = aux_wrong(phase2);
    std::cout << "phase 2, alpha-CROWN within a gap of 0.5: " << wrong3 << " wrong\n";
    check(wrong3 == 0, "alpha-CROWN within optimize_gap is sound");
    std::cout << "phase 2: " << wrong2 << " wrong of " << kNetworks << "\n";
    check(wrong2 == 0, "zonotope-alpha, the backward bound, the dual certificate, fsb and the "
                       "chord rows are sound");
    return test::finish("neuralNetwork verify witness");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
