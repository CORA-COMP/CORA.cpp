// test_neural_network_composite_layer - CORA's composite layer: branches added or concatenated
//
// The block's network computes, at points, the sum or the concatenation of its branches' outputs
// (a skip connection is an empty branch), along any dimension of the branches' output shapes, and a
// batch of zonotopes through it stays enclosing.
//
// CORA: nn/layers/other/nnCompositeLayer.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/nn/neural_network/neural_network.h"
#include "testing.h"

#include <cmath>

using namespace cora;
using test::check;
using Layer = NeuralNetwork::Layer;
using Block = NeuralNetwork::Block;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A branch of one layer on two inputs: W x + b, then the activation.
Block aux_branch(double scale, Activation a) {
    return {{Layer::linear(Tensor({{scale, 1.0}, {0.5, -scale}}), Tensor({0.1, -0.2}))}, {a}};
}

/// The branch's function.
std::vector<double> aux_apply(double scale, Activation a, double x0, double x1) {
    std::vector<double> y{scale * x0 + x1 + 0.1, 0.5 * x0 - scale * x1 - 0.2};
    for (double &v : y) v = a == Activation::relu ? std::max(v, 0.0) : v;
    return y;
}

void adding_and_joining(const std::string &b) {
    const Block one = aux_branch(2.0, Activation::relu),
                two = aux_branch(-1.0, Activation::identity);
    // an add: the sum, the second branch two layers, the third the skip
    Block deep = aux_branch(1.0, Activation::identity);
    deep.layers.push_back(Layer::linear(Tensor({{1.0, 0.0}, {0.0, 3.0}}), Tensor({0.0, 0.0})));
    deep.activations.push_back(Activation::relu);
    const NeuralNetwork sum(
        NeuralNetwork::composite_layer({one, deep, Block{}}, Aggregation::add, 2));
    const double x0 = 0.7, x1 = -0.4;
    const std::vector<double> a = aux_apply(2.0, Activation::relu, x0, x1),
                              d = aux_apply(1.0, Activation::identity, x0, x1);
    check(test::close(sum.evaluate(Tensor({{x0, x1}})),
                      {a[0] + std::max(d[0], 0.0) + x0, a[1] + std::max(3.0 * d[1], 0.0) + x1},
                      1e-12),
          b + ": the sum of three branches");
    // a concat of flat vectors, then along the second dimension of (2, 1) shapes (still flat)
    const NeuralNetwork cat(NeuralNetwork::composite_layer({one, two}, Aggregation::concat, 2));
    const std::vector<double> c = aux_apply(-1.0, Activation::identity, x0, x1);
    check(test::close(cat.evaluate(Tensor({{x0, x1}})), {a[0], a[1], c[0], c[1]}, 1e-12),
          b + ": the concatenation of two branches");
    // outputs (2, 2) joined along dimension 1: rows interleave
    const NeuralNetwork rows(NeuralNetwork::composite_layer(
        {{{Layer::identity(4)}, {Activation::identity}}, {{Layer::identity(4)}, {Activation::exp}}},
        Aggregation::concat, 4, {{2, 2}, {2, 2}}, 1));
    const std::vector<double> y = rows.evaluate(Tensor({{0.0, 1.0, 2.0, 3.0}})).data();
    check(test::close(Tensor::from_data(y, {8}),
                      {0.0, 1.0, 1.0, std::exp(1.0), 2.0, 3.0, std::exp(2.0), std::exp(3.0)},
                      1e-12),
          b + ": a join along the second dimension");
    // a batch of zonotopes through the graph holds the points of its box
    const Tensor c0 = Tensor({{x0, x1}}), G = Tensor::from_data({0.05, 0.0, 0.0, 0.05}, {1, 2, 2});
    const ZonotopeBatch out = sum.evaluate_zonotope_batch(c0, G);
    // a point of the box against the batch's center and reach
    const std::vector<double> y0 = sum.evaluate(Tensor({{x0 + 0.04, x1 - 0.03}})).data(),
                              cc = out.c.data();
    const std::vector<double> reach = out.G.abs().sum(1).data();
    bool inside = true;
    for (std::size_t i = 0; i < 2; ++i)
        inside = inside && std::abs(y0[i] - cc[i]) <= reach[i] + 1e-12;
    check(inside, b + ": a batch of zonotopes encloses a point");
}

void mistakes(const std::string &b) {
    const Block one = aux_branch(2.0, Activation::identity);
    // a branch of one output, and the mistakes that the factory refuses
    const Block wide{{Layer::linear(Tensor({{1.0, 0.0}}), Tensor({0.0}))}, {Activation::identity}};
    check(test::raises([&] { NeuralNetwork::composite_layer({}, Aggregation::add, 2); },
                       Error::wrong_input_in_constructor),
          b + ": no branches");
    check(test::raises([&] { NeuralNetwork::composite_layer({one, wide}, Aggregation::add, 2); },
                       Error::wrong_input_in_constructor),
          b + ": an add of different sizes");
    check(test::raises(
              [&] {
                  NeuralNetwork::composite_layer({one, wide}, Aggregation::concat, 2, {{3}, {1}});
              },
              Error::wrong_input_in_constructor),
          b + ": a shape that does not fit the outputs");
    // shapes that differ where they are not joined, and of different ranks
    check(test::raises(
              [&] {
                  NeuralNetwork::composite_layer({one, one}, Aggregation::concat, 2,
                                                 {{2, 1}, {1, 2}}, 0);
              },
              Error::wrong_input_in_constructor),
          b + ": shapes that differ where they are not joined");
    check(test::raises(
              [&] {
                  NeuralNetwork::composite_layer({one, one}, Aggregation::concat, 2, {{2}, {2, 1}});
              },
              Error::wrong_input_in_constructor),
          b + ": shapes of different ranks");
    // a dimension that is none, a branch with a layer but no activation
    check(test::raises(
              [&] { NeuralNetwork::composite_layer({one, one}, Aggregation::concat, 2, {}, 3); },
              Error::wrong_input_in_constructor),
          b + ": a dimension that is not one");
    check(test::raises(
              [&] {
                  Block bad = one;
                  bad.activations.clear();
                  NeuralNetwork::composite_layer({bad}, Aggregation::add, 2);
              },
              Error::wrong_input_in_constructor),
          b + ": a layer without an activation");
    {
        // a valid call without checks
        ChecksScope off(false);
        check(NeuralNetwork::composite_layer({one, one}, Aggregation::add, 2).layers.size() == 3,
              b + ": a valid call without checks");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        adding_and_joining(b);
        mistakes(b);
    });
    return test::finish("neuralNetwork composite layer");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
