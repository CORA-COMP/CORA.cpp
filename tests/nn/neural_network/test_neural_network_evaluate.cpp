// test_neural_network_evaluate - a network on points, and its zonotope enclosure of the outputs
//
// CORA: nn/@neuralNetwork/evaluate.m
// CORA: nn/@neuralNetwork/neuralNetwork.m
// CORA: nn/layers/nonlinear/nnReLULayer.m
// CORA: nn/layers/nonlinear/nnTanhLayer.m
// CORA: nn/layers/nonlinear/nnSigmoidLayer.m
// CORA: nn/layers/nonlinear/nnSShapeLayer.m
// CORA: nn/@neuralNetwork/evaluate_.m
// PROPERTY: nn/@neuralNetwork/evaluate.m
// CORA authors: Tobias Ladner, Lukas Koller, Sebastian Sigl
// Authors:       Tobias Ladner
// Written:       02-November-2026
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

/// A matrix of the given size with normal entries of standard deviation `sigma`.
Tensor random_matrix(cora::Rng &rng, int64_t rows, int64_t cols, double sigma) {
    std::vector<double> data(rows * cols);
    rng.normal(data.data(), data.size(), sigma);
    return Tensor::from_data(data, {rows, cols});
}

/// A random network with the given layer sizes (input first).
NeuralNetwork random_network(cora::Rng &rng, const std::vector<int64_t> &sizes) {
    std::vector<NeuralNetwork::Layer> layers;
    for (std::size_t k = 1; k < sizes.size(); ++k)
        layers.push_back({random_matrix(rng, sizes[k], sizes[k - 1], 1.0),
                          random_matrix(rng, sizes[k], 1, 0.5)});
    return NeuralNetwork(layers);
}

void points_follow_the_layers(const std::string &b) {
    const NeuralNetwork nn(
        {{Tensor({{1, -1}, {2, 0}}), Tensor({0.0, -1.0})}, {Tensor({{1, 1}}), Tensor({0.5})}});
    const Tensor y = nn.evaluate(Tensor({{1.0, 3.0}, {2.0, 1.0}}));
    // (1, 2): hidden (-1, 1) -> (0, 1) -> 1.5;  (3, 1): hidden (2, 5) -> 7.5
    check(y.shape() == std::vector<int64_t>({1, 2}), b + ": one output per point");
    check(test::close(y, std::vector<double>{1.5, 7.5}), b + ": the outputs of two points");
    check(nn.input_dim() == 2 && nn.output_dim() == 1 && nn.num_layers() == 2, b + ": the sizes");
}

/// Without a ReLU (one layer) the set is mapped exactly.
void an_affine_network_maps_the_set_exactly(const std::string &b) {
    cora::Rng rng(1);
    const NeuralNetwork nn = random_network(rng, {3, 2});
    const Zonotope X = Zonotope::generate_random(rng, 3, std::nullopt, 4);
    const Zonotope Y = nn.evaluate(X);
    const Zonotope want = X.linear_map(nn.layers()[0].W) + nn.layers()[0].b;
    check(test::close(Y.center(), want.center()) && test::close(Y.generators(), want.generators()),
          b + ": W X + b");
}

/// A neuron that keeps its sign is exact: the identity for x > 0, the zero set for x < 0.
void stable_neurons_are_exact(const std::string &b) {
    const Tensor I = Tensor::eye(2), zero = Tensor({0.0, 0.0});
    const NeuralNetwork nn({{I, zero}, {I, zero}});
    const Zonotope positive(Tensor({5.0, 6.0}), Tensor({{1.0, 0.5}, {0.0, 1.0}}));
    const Zonotope negative(Tensor({-5.0, -6.0}), Tensor({{1.0, 0.5}, {0.0, 1.0}}));
    cora::Rng rng(2);
    for (int k = 0; k < 10; ++k) {
        const std::vector<double> d = test::random_direction(rng, 2);
        check(test::close(test::support(nn.evaluate(positive), d), test::support(positive, d)),
              b + ": a positive set passes unchanged");
        check(test::close(test::support(nn.evaluate(negative), d), 0.0),
              b + ": a negative set gives 0");
    }
}

/// One neuron over [-1, 3]: the upper end of the range is exact, the lower one is -0.75.
void a_crossing_neuron_gets_the_tightest_parallelogram(const std::string &b) {
    const NeuralNetwork nn({{Tensor({{1.0}}), Tensor({0.0})}, {Tensor({{1.0}}), Tensor({0.0})}});
    const Interval range = nn.evaluate(Zonotope(Tensor({1.0}), Tensor({{2.0}}))).interval();
    check(test::close(range.sup().data()[0], 3.0), b + ": the upper end is 3");
    check(test::close(range.inf().data()[0], -0.75), b + ": the lower end is -0.75");
}

/// The outputs of every sampled point lie in the enclosure: support along random directions.
void the_enclosure_contains_the_outputs(const std::string &b) {
    cora::Rng rng(3);
    const NeuralNetwork nn = random_network(rng, {3, 6, 5, 2});
    const Zonotope X = Zonotope::generate_random(rng, 3, std::nullopt, 5);
    const Zonotope Y = nn.evaluate(X);
    int violations = 0;
    for (const std::string type : {"standard", "extreme"}) {
        const std::vector<double> y = nn.evaluate(X.rand_point(200, rng, type)).data(); // rows 0, 1
        for (int k = 0; k < 10; ++k) {
            const std::vector<double> d = test::random_direction(rng, 2);
            const double bound = test::support(Y, d);
            for (int j = 0; j < 200; ++j)
                violations += d[0] * y[j] + d[1] * y[200 + j] > bound + 1e-9;
        }
    }
    check(violations == 0, b + ": " + std::to_string(violations) + " outputs left the enclosure");
}

/// A network of the four activations: the points follow the layers, the enclosure holds them.
NeuralNetwork mixed_network(cora::Rng &rng) {
    return NeuralNetwork(
        {{random_matrix(rng, 4, 3, 1.0), random_matrix(rng, 4, 1, 0.5)},
         {random_matrix(rng, 4, 4, 1.0), random_matrix(rng, 4, 1, 0.5)},
         {random_matrix(rng, 3, 4, 1.0), random_matrix(rng, 3, 1, 0.5)},
         {random_matrix(rng, 2, 3, 1.0), random_matrix(rng, 2, 1, 0.5)}},
        {Activation::tanh, Activation::sigmoid, Activation::relu, Activation::identity});
}

void activations_follow_their_functions(const std::string &b) {
    const Tensor one({{1.0}});
    const Tensor x({{0.5}});
    for (const auto &[act, want] : std::vector<std::pair<Activation, double>>{
             {Activation::tanh, std::tanh(0.5)},
             {Activation::sigmoid, 1.0 / (1.0 + std::exp(-0.5))},
             {Activation::relu, 0.5},
             {Activation::identity, 0.5}}) {
        const NeuralNetwork nn({{one, Tensor({0.0})}}, {act});
        check(test::close(nn.evaluate(x).data()[0], want), b + ": an activation at 0.5");
    }
}

/// tanh and sigmoid: the enclosure of a set holds every output, and a degenerate set is a point.
void s_shaped_enclosures_hold_the_outputs(const std::string &b) {
    cora::Rng rng(4);
    const NeuralNetwork nn = mixed_network(rng);
    const Zonotope X = Zonotope::generate_random(rng, 3, std::nullopt, 4);
    const Zonotope Y = nn.evaluate(X);
    int violations = 0;
    for (const std::string type : {"standard", "extreme"}) {
        const std::vector<double> y = nn.evaluate(X.rand_point(200, rng, type)).data();
        for (int k = 0; k < 10; ++k) {
            const std::vector<double> d = test::random_direction(rng, 2);
            const double bound = test::support(Y, d);
            for (int j = 0; j < 200; ++j)
                violations += d[0] * y[j] + d[1] * y[200 + j] > bound + 1e-9;
        }
    }
    check(violations == 0, b + ": " + std::to_string(violations) + " outputs left the enclosure");
    // a set of one point is mapped to the output of the point
    const Tensor p({{0.3}, {-0.2}, {0.1}});
    const Zonotope point = nn.evaluate(Zonotope(p, p.zeros_like({3, 1})));
    check(test::close(point.center(), nn.evaluate(p)), b + ": a point set gives the point");
}

/// evaluate_with_input: the first rows are the input set, the pairs (x, N(x)) of the points lie in
/// the set (the same factor), and the polynomial zonotope keeps the exponents and identifiers.
void the_joint_set_holds_the_pairs(const std::string &b) {
    // the entry (row i, point j) of an array of 300 points
    const auto at = [](int i, int j) { return std::size_t(i) * 300 + std::size_t(j); };
    // a random network and set
    cora::Rng rng(5);
    const NeuralNetwork nn = mixed_network(rng);
    const Zonotope X = Zonotope::generate_random(rng, 3, std::nullopt, 4);
    const Zonotope XY = nn.evaluate_with_input(X);
    check(XY.dim() == 5, b + ": the pairs have dimension in + out");
    // the part of the input is X itself
    const Interval hull = XY.interval();
    check(test::close(hull.inf().slice(0, 0, 3), X.interval().inf()) &&
              test::close(hull.sup().slice(0, 0, 3), X.interval().sup()),
          b + ": the input part is the input set");
    // a pair (x, N(x)) of a point x of X lies in the set: the support along random directions
    int violations = 0;
    const Tensor x = X.rand_point(300, rng, "extreme");
    const Tensor y = nn.evaluate(x);
    const std::vector<double> xs = x.data(), ys = y.data();
    for (int k = 0; k < 20; ++k) {
        const std::vector<double> d = test::random_direction(rng, 5);
        const double bound = test::support(XY, d);
        for (int j = 0; j < 300; ++j) {
            double value = 0;
            for (int i = 0; i < 3; ++i) value += d[std::size_t(i)] * xs[at(i, j)];
            for (int i = 0; i < 2; ++i) value += d[std::size_t(3) + std::size_t(i)] * ys[at(i, j)];
            violations += value > bound + 1e-9;
        }
    }
    check(violations == 0, b + ": " + std::to_string(violations) + " pairs left the set");
    // the polynomial zonotope: factors and exponents stay, errors as independent or new factors
    const PolyZonotope P(X);
    const PolyZonotope in_gi = nn.evaluate_with_input(P);
    check(in_gi.ids() == P.ids() && in_gi.num_dependent() == P.num_dependent() &&
              in_gi.exponents() == P.exponents(),
          b + ": the factors and the exponents stay");
    const PolyZonotope as_factors = nn.evaluate_with_input(P, {.add_approx_error_to_gi = false});
    // the errors as new factors leave no independent generators
    check(as_factors.num_factors() > P.num_factors() &&
              as_factors.independent_generators().shape()[1] == 0,
          b + ": the errors become new factors");
    // both enclose the pairs: the zonotope enclosure of each holds them
    for (const PolyZonotope *Q : {&in_gi, &as_factors}) {
        const Zonotope QZ = Q->zonotope();
        int out = 0;
        for (int k = 0; k < 20; ++k) {
            const std::vector<double> d = test::random_direction(rng, 5);
            const double bound = test::support(QZ, d);
            // the value of the pair j along d
            for (int j = 0; j < 300; ++j) {
                double value = 0;
                for (int i = 0; i < 3; ++i) value += d[std::size_t(i)] * xs[at(i, j)];
                for (int i = 0; i < 2; ++i)
                    value += d[std::size_t(3) + std::size_t(i)] * ys[at(i, j)];
                out += value > bound + 1e-9;
            }
        }
        check(out == 0, b + ": " + std::to_string(out) + " pairs left the polynomial zonotope");
    }
}

void wrong_arguments_are_described(const std::string &b) {
    // layers and sets that do not fit are refused
    const Tensor W = Tensor({{1, 2}, {3, 4}}), bias = Tensor({0.0, 0.0});
    check(test::throws([&] { NeuralNetwork(std::vector<NeuralNetwork::Layer>{}); }),
          b + ": no layers");
    check(test::throws([&] { NeuralNetwork({{W, Tensor({0.0, 0.0, 0.0})}}); }),
          b + ": a bias of another size");
    check(test::throws([&] { NeuralNetwork({{W, bias}, {Tensor({{1, 2, 3}}), Tensor({0.0})}}); }),
          b + ": layers that do not fit");
    check(test::throws([&] { NeuralNetwork({{W, bias}}, {Activation::relu, Activation::tanh}); }),
          b + ": one activation per layer");
    const NeuralNetwork nn({{W, bias}});
    check(test::throws([&] { nn.evaluate(Tensor({1.0, 2.0, 3.0})); }),
          b + ": a point of another dimension");
    // a set of another dimension
    check(test::throws([&] { nn.evaluate(Zonotope(Tensor({1.0}), Tensor({{1.0}}))); }),
          b + ": a set of another dimension");
}

/// Points of the wrong dimension are refused; with the checks off the outputs are the same.
void checks_and_their_switch() {
    const NeuralNetwork net({{Tensor({{1.0, -1.0}, {0.5, 2.0}}), Tensor({{0.0}, {1.0}})}});
    check(test::raises([&] { net.evaluate(Tensor({{1.0}, {0.0}, {0.0}})); },
                       Error::dimension_mismatch),
          "evaluate: a point of three dimensions");
    const Tensor x({{1.0}, {2.0}});
    const Tensor on = net.evaluate(x);
    ChecksScope off(false);
    check(test::close(net.evaluate(x), on, 0), "evaluate: the same with the checks off");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        points_follow_the_layers(b);
        an_affine_network_maps_the_set_exactly(b);
        stable_neurons_are_exact(b);
        a_crossing_neuron_gets_the_tightest_parallelogram(b);
        the_enclosure_contains_the_outputs(b);
        activations_follow_their_functions(b);
        s_shaped_enclosures_hold_the_outputs(b);
        the_joint_set_holds_the_pairs(b);
        wrong_arguments_are_described(b);
    });
    checks_and_their_switch();
    return test::finish("neuralNetwork evaluate");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
