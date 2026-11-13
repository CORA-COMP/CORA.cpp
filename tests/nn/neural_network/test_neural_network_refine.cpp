// test_neural_network_refine - approx_errors, refinable layers and refine on the orders
//
// After an evaluation each neuron has the radius of its approximation error. refine raises the
// orders by one: all refinable neurons, the layer of the largest norm, or the neurons above gamma
// times the largest heuristic in its layer, up to max_order. A property test on the choices.
//
// CORA: nn/@neuralNetwork/refine.m
// CORA: nn/@neuralNetwork/getRefinableLayers.m
// PROPERTY: nn/@neuralNetwork/refine.m
// CORA authors: Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       13-November-2026
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

/// approx_errors gives the radius per neuron, refine raises the orders its heuristic names.
void refine_raises_the_orders_of_the_heuristic(const std::string &b) {
    const NeuralNetwork nn({{Tensor({{1.0, -1.0}, {2.0, 1.0}}), Tensor({0.0, 0.1})},
                            {Tensor({{1.0, 0.5}, {-0.5, 1.0}}), Tensor({0.0, 0.0})},
                            {Tensor({{1.0, 1.0}}), Tensor({0.0})}},
                           {Activation::relu, Activation::tanh, Activation::identity});
    // heuristics drawn from a fixed seed: the layer of the largest norm is raised
    Rng rng(1);
    std::vector<double> noise(4);
    rng.normal(noise.data(), noise.size(), 1.0);
    const std::vector<std::vector<double>> drawn{
        {std::abs(noise[0]), std::abs(noise[1])}, {std::abs(noise[2]), std::abs(noise[3])}, {}};
    const auto by_draw = nn.refine({}, drawn);
    const std::size_t pick =
        std::hypot(drawn[0][0], drawn[0][1]) >= std::hypot(drawn[1][0], drawn[1][1]) ? 0 : 1;
    check(by_draw[pick][0] == 2 && by_draw[1 - pick][0] == 1, b + ": a drawn heuristic");
    check(nn.refinable_layers() == std::vector<int64_t>({0, 1}), b + ": the refinable layers");
    const PolyZonotope X(Tensor({{0.0}, {0.0}}), Tensor({{1.0, 0.0}, {0.0, 1.0}}),
                         Tensor::zeros({2, 0}), {{1, 0}, {0, 1}}, {1, 2});
    const auto errors = nn.approx_errors(X);
    check(errors.size() == 3 && errors[0].size() == 2 && errors[1].size() == 2 && errors[2].empty(),
          b + ": an error per neuron of a refinable layer");
    // layer: the layer of the larger norm, all its neurons by one
    const auto norm = [&](std::size_t k) { return std::hypot(errors[k][0], errors[k][1]); };
    const std::size_t larger = norm(0) >= norm(1) ? 0 : 1;
    const auto by_layer = nn.refine({}, errors);
    check(by_layer[larger] == std::vector<int>({2, 2}) &&
              by_layer[1 - larger] == std::vector<int>({1, 1}) &&
              by_layer[2] == std::vector<int>({1}),
          b + ": layer refines the layer of the largest norm");
    // all: every refinable neuron, up to max_order
    auto all = nn.refine({}, errors, {.max_order = 2, .type = RefineType::all});
    all = nn.refine(all, errors, {.max_order = 2, .type = RefineType::all});
    check(all[0] == std::vector<int>({2, 2}) && all[1] == std::vector<int>({2, 2}) &&
              all[2] == std::vector<int>({1}),
          b + ": all stops at max_order");
    // neuron: only the neurons above gamma times the largest, in its layer
    const auto by_neuron = nn.refine({}, {{1.0, 0.05}, {0.5, 0.4}}, {.type = RefineType::neuron});
    check(by_neuron[0] == std::vector<int>({2, 1}) && by_neuron[1] == std::vector<int>({1, 1}),
          b + ": neuron raises the neurons above gamma times the largest");
    // layer_bias: the later layer counts less
    const auto biased =
        nn.refine({}, {{1.0, 1.0}, {1.5, 1.5}}, {.heuristic = RefineHeuristic::layer_bias});
    check(biased[0] == std::vector<int>({2, 2}) && biased[1] == std::vector<int>({1, 1}),
          b + ": layer_bias favours earlier layers");
}

/// A max_order below 1 is refused.
void wrong_arguments_are_described(const std::string &b) {
    const NeuralNetwork nn({{Tensor({{1.0}}), Tensor({0.0})}}, {Activation::sigmoid});
    check(test::raises([&] { nn.refine({}, {{1.0}}, {.max_order = 0}); }, Error::wrong_value),
          b + ": max_order below 1");
}

/// The same orders with the checks off.
void checks_and_their_switch() {
    const NeuralNetwork nn({{Tensor({{1.0}}), Tensor({0.0})}}, {Activation::relu});
    const auto on = nn.refine({}, {{1.0}});
    ChecksScope off(false);
    check(nn.refine({}, {{1.0}}) == on, "refine: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        refine_raises_the_orders_of_the_heuristic(b);
        wrong_arguments_are_described(b);
    });
    checks_and_their_switch();
    return test::finish("neuralNetwork refine");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
