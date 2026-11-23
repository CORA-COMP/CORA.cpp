// test_neural_network_group_sort - CORA's group sort layer at points and its gradient
//
// A group of consecutive neurons is sorted ascending (the last group may be shorter), the layer
// is a permutation, and its gradient returns each sorted position's entry to its input (checked by
// finite differences). Sets have no group sort, as in CORA.
//
// CORA: nn/layers/nonlinear/nnGroupSortLayer.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/nn/neural_network/neural_network.h"
#include "nn/neural_network/private/priv_layers.h"
#include "testing.h"

#include <algorithm>
#include <cmath>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The values with every group of g consecutive entries sorted ascending.
std::vector<double> aux_sorted(std::vector<double> x, int64_t g) {
    for (std::size_t i = 0; i < x.size(); i += std::size_t(g))
        std::sort(x.begin() + std::ptrdiff_t(i),
                  x.begin() + std::ptrdiff_t(std::min(x.size(), i + std::size_t(g))));
    return x;
}

void sorting(const std::string &b) {
    const std::vector<double> x{0.5, -1.0, 2.0, 0.1, 0.1, -3.0, 4.0};
    // the layer on points for groups that divide, that do not, and that exceed the layer
    for (const int64_t g : {1, 2, 3, 7, 9}) {
        const NeuralNetwork nn(NeuralNetwork::group_sort_layer(7, g));
        check(test::close(nn.evaluate(Tensor::from_data(x, {1, 7})), aux_sorted(x, g), 0.0),
              b + ": groups of " + std::to_string(g));
    }
    // two points at once; rows of any shape for the function itself
    const Tensor two = Tensor::from_data({3.0, 1.0, 2.0, 0.0, 5.0, 4.0}, {2, 3});
    check(test::close(priv_group_sort(two, 3), {1.0, 2.0, 3.0, 0.0, 4.0, 5.0}, 0.0),
          b + ": a batch of rows");
    // CORA's layer takes a bias: a layer before it
    const NeuralNetwork chain(
        {NeuralNetwork::Layer::linear(Tensor({{1.0, 0.0}, {0.0, 1.0}}), Tensor({0.0, -1.0})),
         NeuralNetwork::group_sort_layer(2, 2, 0).layers[0]},
        {Activation::identity, Activation::group_sort});
    check(test::close(chain.evaluate(Tensor({{0.5, 0.25}})), {-0.75, 0.5}, 0.0),
          b + ": after a linear layer");
    check(test::raises([&] { priv_group_sort(two, 0); }, Error::wrong_value),
          b + ": a group of no neurons");
    check(
        test::raises([] { NeuralNetwork::group_sort_layer(0); }, Error::wrong_input_in_constructor),
        b + ": a layer of no neurons");
}

void gradient(const std::string &b) {
    // loss = sum w_i y_i; the gradient at the input against finite differences
    const std::vector<double> x{0.5, -1.0, 2.0, 0.1, 0.7, -3.0, 4.0},
        w{1.0, 2.0, -1.0, 0.5, 3.0, -2.0, 0.25};
    for (const int64_t g : {2, 3, 4}) {
        // the gradient of the network, and the finite differences of the loss
        const NeuralNetwork nn(NeuralNetwork::group_sort_layer(7, g));
        const Tensor grad = nn.backprop(Tensor::from_data(x, {1, 7}), Tensor::from_data(w, {1, 7}));
        const auto loss = [&](const std::vector<double> &p) {
            const std::vector<double> y = nn.evaluate(Tensor::from_data(p, {1, 7})).data();
            double s = 0;
            for (std::size_t i = 0; i < 7; ++i) s += w[i] * y[i];
            return s;
        };
        // central differences
        std::vector<double> numeric;
        for (std::size_t i = 0; i < 7; ++i) {
            std::vector<double> up = x, down = x;
            up[i] += 1e-6;
            down[i] -= 1e-6;
            numeric.push_back((loss(up) - loss(down)) / 2e-6);
        }
        check(test::close(grad, numeric, 1e-6),
              b + ": the gradient of groups of " + std::to_string(g));
    }
}

void sets(const std::string &b) {
    // a set of two neurons
    const NeuralNetwork nn(NeuralNetwork::group_sort_layer(2));
    const Zonotope X(Tensor({0.0, 1.0}), Tensor({{0.1, 0.0}, {0.0, 0.1}}));
    // a group sort has no line, no derivative bounds, no sensitivity, and no verification
    check(test::raises([&] { nn.evaluate(X); }, Error::not_supported), b + ": evaluate on a set");
    check(test::raises(
              [&] {
                  nn.evaluate_zonotope_batch(Tensor({{0.0, 1.0}}),
                                             Tensor::from_data({0.1, 0.0}, {1, 1, 2}));
              },
              Error::not_supported),
          b + ": a batch of zonotopes");
    check(test::raises(
              [&] {
                  nn.backprop_interval_batch(Tensor({{0.0, 1.0}}), Tensor({{0.0, 1.0}}),
                                             Tensor({{1.0, 1.0}}), Tensor({{1.0, 1.0}}));
              },
              Error::not_supported),
          b + ": the bounds of the gradient");
    check(test::raises([&] { nn.sensitivity(Tensor({0.0, 1.0})); }, Error::not_supported),
          b + ": the Jacobian");
    check(test::raises([&] { nn.verify(NnProperty{}, 1.0); }, Error::not_supported),
          b + ": verify");
    {
        // a valid call without checks
        ChecksScope off(false);
        check(priv_group_sort(Tensor({{2.0, 1.0}}), 2).numel() == 2,
              b + ": a valid call without checks");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        sorting(b);
        gradient(b);
        sets(b);
    });
    return test::finish("neuralNetwork group sort");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
