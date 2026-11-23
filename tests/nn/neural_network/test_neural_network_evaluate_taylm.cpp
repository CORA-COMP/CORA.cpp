// test_neural_network_evaluate_taylm - the network on Taylor models
//
// A Taylor model per input goes through the layers (the map of each layer, then per neuron the
// polynomial of its bounds with the error as the remainder). The tests check exact cases by hand
// (an identity network, a stable ReLU) and that the outputs of sampled points lie in the bounds
// (orders 1 to 3, both methods). CORALean serves no op for it (docs/oracle-requests.md), so these
// are property tests.
//
// CORA: nn/layers/nonlinear/@nnActivationLayer/evaluateTaylm.m
// PROPERTY: nn/layers/nonlinear/@nnActivationLayer/evaluateTaylm.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/global/rng.h"
#include "cora/nn/neural_network/neural_network.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {


/// A two-layer network 2 -> 3 -> 2 with the activation a after the first layer.
NeuralNetwork aux_network(Activation a) {
    return NeuralNetwork(
        {{Tensor({{1.0, -2.0}, {0.5, 1.5}, {-1.0, 1.0}}), Tensor({0.1, -0.2, 0.3})},
         {Tensor({{1.0, 1.0, -1.0}, {0.5, -1.0, 2.0}}), Tensor({0.0, 0.2})}},
        {a, Activation::identity});
}

/// The models of the box [lo, hi]^2 and the outputs of the network at the points of its grid
/// lie in the bounds of the models of the outputs.
void taylm_contains(const std::string &b, Activation a, const NnOptions &options, double lo,
                    double hi) {
    const NeuralNetwork nn = aux_network(a);
    const std::vector<Taylm> y = nn.evaluate_taylm(
        Taylm::from_interval(Interval(Tensor({lo, lo}), Tensor({hi, hi}))), options);
    check(y.size() == 2, b + ": one model per output");
    bool inside = true;
    // the grid of the box and 20 random points
    Rng rng(4);
    std::vector<double> random(40);
    rng.uniform(random.data(), random.size(), lo, hi);
    for (int i = 0; i < 9 + 20; ++i)
        for (int j = 0; j <= (i < 9 ? 8 : 0); ++j) {
            const double x = i < 9 ? lo + (hi - lo) * i / 8.0 : random[2 * std::size_t(i - 9)],
                         z = i < 9 ? lo + (hi - lo) * j / 8.0 : random[2 * std::size_t(i - 9) + 1];
            const Tensor out = nn.evaluate(Tensor({{x, z}}));
            const std::vector<double> v = out.data();
            for (std::size_t k = 0; k < 2; ++k) {
                const std::array<double, 2> r = y[k].bounds();
                inside = inside && r[0] <= v[k] + 1e-12 && v[k] - 1e-12 <= r[1];
            }
        }
    check(inside, b + ": the sampled outputs lie in the bounds");
}

void taylm(const std::string &b) {
    // an identity network is exact: the bounds of the image of the box
    const NeuralNetwork lin({{Tensor({{1.0, 2.0}}), Tensor({0.5})}}, {Activation::identity});
    const std::vector<Taylm> y =
        lin.evaluate_taylm(Taylm::from_interval(Interval(Tensor({0.0, -1.0}), Tensor({1.0, 1.0}))));
    check(test::close(y[0].bounds()[0], -1.5, 1e-12) && test::close(y[0].bounds()[1], 3.5, 1e-12),
          b + ": the identity network is the image of the box");
    // a ReLU that is active over the whole box keeps the line
    const NeuralNetwork relu({{Tensor({{1.0}}), Tensor({3.0})}}, {Activation::relu});
    const std::vector<Taylm> r = relu.evaluate_taylm({Taylm::from_range(0.0, 1.0, 1)});
    check(test::close(r[0].bounds()[0], 3.0, 1e-12) && test::close(r[0].bounds()[1], 4.0, 1e-12),
          b + ": a stable ReLU is exact");
    // the sampled outputs lie in the models, per activation, method and order
    NnOptions singh, regression, order3;
    regression.poly_method = PolyMethod::regression;
    order3.orders = {{3}};
    order3.poly_method = PolyMethod::regression;
    taylm_contains(b + " relu", Activation::relu, singh, -1.0, 1.0);
    taylm_contains(b + " sigmoid regression", Activation::sigmoid, regression, -1.0, 1.0);
    taylm_contains(b + " tanh order 3", Activation::tanh, order3, -0.5, 1.0);
    NnOptions per_neuron;
    per_neuron.orders = {{1, 2, 3}};
    per_neuron.poly_method = PolyMethod::regression;
    taylm_contains(b + " tanh per neuron", Activation::tanh, per_neuron, -1.0, 0.5);
    // a list of another size takes its largest order
    NnOptions other;
    other.orders = {{1, 2}};
    taylm_contains(b + " relu largest order", Activation::relu, other, -1.0, 1.0);
    // the checks
    check(test::raises([&] { lin.evaluate_taylm({Taylm::constant(0.0, 1)}); },
                       Error::dimension_mismatch),
          b + ": a wrong number of models");
    check(test::raises(
              [&] { lin.evaluate_taylm({Taylm::constant(0.0, 1), Taylm::constant(0.0, 2)}); },
              Error::dimension_mismatch),
          b + ": models over different variables");
    const NeuralNetwork graph({NeuralNetwork::Layer::linear(Tensor({{1.0}}), Tensor({0.0})),
                               [] {
                                   NeuralNetwork::Layer L =
                                       NeuralNetwork::Layer::linear(Tensor({{1.0}}), Tensor({0.0}));
                                   L.source = -1;
                                   L.residuals = {{0, nullptr}};
                                   return L;
                               }()},
                              {Activation::identity, Activation::identity});
    check(test::raises([&] { graph.evaluate_taylm({Taylm::constant(0.0, 1)}); },
                       Error::not_supported),
          b + ": a graph");
    {
        // a valid call without checks
        ChecksScope off(false);
        check(lin.evaluate_taylm(
                     Taylm::from_interval(Interval(Tensor({0.0, 0.0}), Tensor({1.0, 1.0}))))
                      .size() == 1,
              b + ": a valid call without checks");
    }
}


} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { taylm(b); });
    return test::finish("neuralNetwork evaluate on Taylor models");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
