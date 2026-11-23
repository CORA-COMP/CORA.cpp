// test_neural_network_backprop - the gradient of a network, its bounds, and the bounds of layers
//
// backprop is the chain rule (the Jacobian of sensitivity times the gradient), the interval version
// is the same on a point box and exact through an affine layer, and propagate_bounds carries
// bounds to the next activation layer.
//
// CORA: nn/@neuralNetwork/backprop.m
// CORA: nn/@neuralNetwork/backpropIntervalBatch.m
// CORA: nn/@neuralNetwork/propagateBounds.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/nn/neural_network/neural_network.h"
#include "testing.h"

#include <cmath>
#include <limits>

using namespace cora;
using test::check;
using Layer = NeuralNetwork::Layer;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Three layers on two inputs with the activations a, relu and the identity.
NeuralNetwork aux_network(Activation a) {
    return NeuralNetwork(
        {{Tensor({{1.0, -0.5}, {0.3, 0.8}, {-0.7, 0.4}}), Tensor({0.2, -0.1, 0.0})},
         {Tensor({{0.5, 1.0, -1.0}, {1.0, -0.2, 0.3}}), Tensor({0.1, 0.0})},
         {Tensor({{1.0, 2.0}}), Tensor({0.5})}},
        {a, Activation::relu, Activation::identity});
}

void gradients(const std::string &b) {
    for (const Activation a : {Activation::sigmoid, Activation::tanh, Activation::square,
                               Activation::exp, Activation::relu, Activation::reciprocal}) {
        const NeuralNetwork nn = aux_network(a);
        const Tensor x({{0.4, 0.9}, {-0.6, 0.3}}), g({{2.0}, {-1.5}});
        // the Jacobian (1, 2) of each point times its gradient
        const Tensor got = nn.backprop(x, g);
        const std::vector<double> j0 = nn.sensitivity(Tensor({0.4, 0.9})).data(),
                                  j1 = nn.sensitivity(Tensor({-0.6, 0.3})).data();
        check(test::close(got,
                          std::vector<double>{2.0 * j0[0], 2.0 * j0[1], -1.5 * j1[0], -1.5 * j1[1]},
                          1e-12),
              b + ": backprop is the Jacobian times the gradient (activation " +
                  std::to_string(int(a)) + ")");
        // on a point box the interval version is the same
        const auto [gl, gu] = nn.backprop_interval_batch(x, x, g, g);
        check(test::close(gl, got.data(), 1e-12) && test::close(gu, got.data(), 1e-12),
              b + ": the interval version on a point box (activation " + std::to_string(int(a)) +
                  ")");
    }
    // a box through affine layers: W^T g -+ |W|^T r, exactly
    const NeuralNetwork lin({{Tensor({{1.0, -2.0}}), Tensor({0.0})}}, {Activation::identity});
    const auto [lo, hi] = lin.backprop_interval_batch(Tensor({{0.0, 0.0}}), Tensor({{1.0, 1.0}}),
                                                      Tensor({{1.5}}), Tensor({{2.5}}));
    check(test::close(lo, std::vector<double>{2.0 - 0.5, -4.0 - 1.0}, 1e-12) &&
              test::close(hi, std::vector<double>{2.0 + 0.5, -4.0 + 1.0}, 1e-12),
          b + ": the bounds of the gradient through an affine layer");
    // the layers other than a chain, a product, a wrong size
    const NeuralNetwork product({{Tensor({{1.0}, {1.0}}), Tensor({0.0, 0.0})}},
                                {Activation::product});
    check(test::raises(
              [&] {
                  product.backprop_interval_batch(Tensor({{0.0}}), Tensor({{1.0}}),
                                                  Tensor({{1.0, 1.0}}), Tensor({{1.0, 1.0}}));
              },
              Error::not_supported),
          b + ": a product has no box image");
    check(test::raises(
              [&] { aux_network(Activation::tanh).backprop(Tensor({{1.0}}), Tensor({{1.0}})); },
              Error::dimension_mismatch),
          b + ": a point of the wrong size");
    const NeuralNetwork graph(NeuralNetwork::softmax_layer(2));
    check(test::raises([&] { graph.backprop(Tensor({{1.0, 2.0}}), Tensor({{1.0, 1.0}})); },
                       Error::not_supported),
          b + ": a graph of layers");
    // a product layer's gradient at a point
    check(test::close(product.backprop(Tensor({{2.0}}), Tensor({{1.0, 1.0}})),
                      std::vector<double>{4.0}),
          b + ": the gradient of a product");
}

/// A layer merged away entirely has no neurons: its gradient is the empty row, the one before it
/// gets zero.
void empty_layer(const std::string &b) {
    const Tensor W1({{1.0, -0.5}, {0.3, 0.8}}), b1({0.1, 0.2});
    const NeuralNetwork nn({{W1, b1},
                            {Tensor::zeros({0, 2}), Tensor::zeros({0})},
                            {Tensor::zeros({1, 0}), Tensor({0.5})}},
                           {Activation::relu, Activation::relu, Activation::identity});
    // the gradient and its bounds at two points, the output
    const Tensor x({{0.4, 0.9}, {-0.6, 0.3}}), g({{2.0}, {-1.5}});
    check(test::close(nn.backprop(x, g), std::vector<double>(4, 0.0), 0.0),
          b + ": the gradient through a layer without neurons is 0");
    const auto [gl, gu] = nn.backprop_interval_batch(x, x, g, g);
    check(test::close(gl, std::vector<double>(4, 0.0), 0.0) &&
              test::close(gu, std::vector<double>(4, 0.0), 0.0),
          b + ": the gradient bounds through a layer without neurons are 0");
    check(test::close(nn.evaluate(x), std::vector<double>{0.5, 0.5}, 0.0),
          b + ": a layer without neurons evaluates to the bias");
}

void bounds(const std::string &b) {
    const NeuralNetwork nn = aux_network(Activation::relu);
    const double nan = std::numeric_limits<double>::quiet_NaN();
    std::vector<NnBounds> none(3), have = none;
    have[0] = {Tensor({-1.0, 0.0, 1.0}), Tensor({1.0, 2.0, 3.0})};
    // relu of the boxes [0, 1] x [0, 2] x [1, 3] through W1 and b1 reach the second activation
    const std::vector<NnBounds> out = nn.propagate_bounds(have, 0);
    check(test::close(out[1].l, std::vector<double>{-2.9, -0.1}, 1e-12) &&
              test::close(out[1].u, std::vector<double>{1.6, 1.9}, 1e-12),
          b + ": the interval image of the layers");
    // stored bounds are intersected
    have[1] = {Tensor({0.0, 0.0}), Tensor({1.0, 5.0})};
    const std::vector<NnBounds> met = nn.propagate_bounds(have, 0);
    check(test::close(met[1].u, std::vector<double>{1.0, 1.9}, 1e-12) &&
              test::close(met[1].l, std::vector<double>{0.0, 0.0}, 1e-12),
          b + ": the stored bounds are intersected");
    // nothing without bounds, with a NaN, or at an identity layer
    check(!nn.propagate_bounds(none, 0)[1].l.defined(), b + ": no bounds, nothing propagates");
    have[0].l = Tensor({nan, 0.0, 1.0});
    check(test::close(nn.propagate_bounds(have, 0)[1].u, have[1].u.data()),
          b + ": NaN bounds propagate nothing");
    check(!nn.propagate_bounds(have, 2)[2].l.defined(), b + ": an identity layer");
    check(test::raises([&] { nn.propagate_bounds(none, 5); }, Error::wrong_value),
          b + ": a layer out of range");
    {
        // a valid call without checks
        ChecksScope off(false);
        check(nn.propagate_bounds(none, 0).size() == 3, b + ": a valid call without checks");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        gradients(b);
        empty_layer(b);
        bounds(b);
    });
    return test::finish("neuralNetwork backprop");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
