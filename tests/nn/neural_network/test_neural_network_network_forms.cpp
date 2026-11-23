// test_neural_network_network_forms - the order of a network's inputs and its normal form
//
// The inputs in order, along a snake and by the sensitivity; the merged layers compute the same
// function with an activation after each linear layer.
//
// CORA: nn/@neuralNetwork/getInputNeuronOrder.m
// CORA: nn/@neuralNetwork/getNormalForm.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/nn/neural_network/neural_network.h"
#include "testing.h"

using namespace cora;
using test::check;
using Layer = NeuralNetwork::Layer;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

void forms(const std::string &b) {
    // the order of the inputs
    const NeuralNetwork nn({{Tensor({{3.0, 0.0, 1.0, -2.0, 0.0, 0.5}}), Tensor({0.0})}},
                           {Activation::identity});
    const Tensor x({0.0, 0.0, 0.0, 0.0, 0.0, 0.0});
    check(nn.input_neuron_order("in-order", x) == std::vector<int64_t>({0, 1, 2, 3, 4, 5}),
          b + ": in order");
    check(nn.input_neuron_order("in-order", x, {1, 2, 3}) ==
              std::vector<int64_t>({0, 1, 2, 3, 4, 5}),
          b + ": in order of an image");
    check(nn.input_neuron_order("snake", x, {1, 2, 3}) == std::vector<int64_t>({0, 1, 2, 5, 4, 3}),
          b + ": the snake of 2 x 3");
    // the least sensitive input first, stable
    check(nn.input_neuron_order("sensitivity", x) == std::vector<int64_t>({1, 4, 5, 2, 3, 0}),
          b + ": by the sensitivity");
    // two channels of three pixels: the channels' mean per pixel, (3 + 1 + ...) / 2
    check(nn.input_neuron_order("sensitivity", x, {2, 1, 3}) == std::vector<int64_t>({1, 2, 0}),
          b + ": by the sensitivity of the pixels");
    check(test::raises([&] { nn.input_neuron_order("snake", x); }, Error::wrong_value),
          b + ": a snake needs an image");
    check(test::raises([&] { nn.input_neuron_order("zigzag", x); }, Error::wrong_value),
          b + ": an unknown method");
    check(test::raises([&] { nn.input_neuron_order("snake", x, {1, 2, 2}); },
                       Error::dimension_mismatch),
          b + ": an image of the wrong size");
    // the snake of 3 x 3 spirals in
    const NeuralNetwork nine({{Tensor::zeros({1, 9}), Tensor({0.0})}}, {Activation::identity});
    check(nine.input_neuron_order("snake", Tensor::zeros({9}), {1, 3, 3}) ==
              std::vector<int64_t>({0, 1, 2, 5, 8, 7, 6, 3, 4}),
          b + ": the snake of 3 x 3 turns in");
    // the normal form: the linear layers in front of an activation are merged
    const NeuralNetwork deep({Layer::linear(Tensor({{1.0, 2.0}, {0.5, -1.0}}), Tensor({0.1, 0.2})),
                              Layer::linear(Tensor({{1.0, 1.0}, {2.0, 0.0}}), Tensor({0.0, -0.3})),
                              Layer::linear(Tensor({{1.0, -1.0}, {0.5, 0.5}}), Tensor({0.4, 0.0})),
                              Layer::identity(2),
                              Layer::linear(Tensor({{2.0, 1.0}}), Tensor({1.0}))},
                             {Activation::identity, Activation::relu, Activation::identity,
                              Activation::tanh, Activation::identity});
    const NeuralNetwork normal = deep.normal_form();
    check(normal.num_layers() == 3 && normal.activations()[0] == Activation::relu &&
              normal.activations()[1] == Activation::tanh &&
              normal.activations()[2] == Activation::identity,
          b + ": the layers are merged into the normal form");
    const Tensor pts({{0.3, -0.7}, {1.2, 0.4}, {-2.0, 1.0}});
    check(test::close(normal.evaluate(pts), deep.evaluate(pts).data(), 1e-12),
          b + ": the normal form is the same function");
    // a network in normal form stays, and a graph is refused
    check(normal.normal_form().num_layers() == 3, b + ": a normal form is its own normal form");
    check(test::raises([] { NeuralNetwork(NeuralNetwork::softmax_layer(2)).normal_form(); },
                       Error::not_supported),
          b + ": a graph has no normal form");
    {
        // a valid call without checks
        ChecksScope off(false);
        check(normal.normal_form().num_layers() == 3, b + ": a valid call without checks");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { forms(b); });
    return test::finish("neuralNetwork network forms");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
