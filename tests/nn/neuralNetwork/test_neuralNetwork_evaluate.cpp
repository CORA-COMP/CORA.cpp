// test_neuralNetwork_evaluate - a network on points, and its zonotope enclosure of the outputs

#include "global/rng.h"
#include "nn/neuralNetwork/neuralNetwork.h"
#include "testing.h"

using namespace cora;
using test::check;

namespace {

/// A matrix of the given size with normal entries of standard deviation `sigma`.
Tensor randomMatrix(cora::Rng &rng, int64_t rows, int64_t cols, double sigma) {
    std::vector<double> data(rows * cols);
    rng.normal(data.data(), data.size(), sigma);
    return Tensor::fromData(data, {rows, cols});
}

/// A random network with the given layer sizes (input first).
NeuralNetwork randomNetwork(cora::Rng &rng, const std::vector<int64_t> &sizes) {
    std::vector<NeuralNetwork::Layer> layers;
    for (std::size_t k = 1; k < sizes.size(); ++k)
        layers.push_back({randomMatrix(rng, sizes[k], sizes[k - 1], 1.0),
                          randomMatrix(rng, sizes[k], 1, 0.5)});
    return NeuralNetwork(layers);
}

void points_follow_the_layers(const std::string &b) {
    const NeuralNetwork nn({{Tensor({{1, -1}, {2, 0}}), Tensor({0.0, -1.0})},
                            {Tensor({{1, 1}}), Tensor({0.5})}});
    const Tensor y = nn.evaluate(Tensor({{1.0, 3.0}, {2.0, 1.0}}));
    // (1, 2): hidden (-1, 1) -> (0, 1) -> 1.5;  (3, 1): hidden (2, 5) -> 7.5
    check(y.shape() == std::vector<int64_t>({1, 2}), b + ": one output per point");
    check(test::close(y, std::vector<double>{1.5, 7.5}), b + ": the outputs of two points");
    check(nn.inputDim() == 2 && nn.outputDim() == 1 && nn.numLayers() == 2, b + ": the sizes");
}

/// Without a ReLU (one layer) the set is mapped exactly.
void an_affine_network_maps_the_set_exactly(const std::string &b) {
    cora::Rng rng(1);
    const NeuralNetwork nn = randomNetwork(rng, {3, 2});
    const Zonotope X = Zonotope::generateRandom(3, 4, rng);
    const Zonotope Y = nn.evaluate(X);
    const Zonotope want = X.mtimes(nn.layers()[0].W) + nn.layers()[0].b;
    check(test::close(Y.c, want.c) && test::close(Y.G, want.G), b + ": W X + b");
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
        check(test::close(test::support(nn.evaluate(negative), d), 0.0), b + ": a negative set gives 0");
    }
}

/// One neuron over [-1, 3]: the upper end of the range is exact, the lower one is -0.75.
void a_crossing_neuron_gets_the_tightest_parallelogram(const std::string &b) {
    const NeuralNetwork nn({{Tensor({{1.0}}), Tensor({0.0})}, {Tensor({{1.0}}), Tensor({0.0})}});
    const Interval range = nn.evaluate(Zonotope(Tensor({1.0}), Tensor({{2.0}}))).interval();
    check(test::close(range.sup.data()[0], 3.0), b + ": the upper end is 3");
    check(test::close(range.inf.data()[0], -0.75), b + ": the lower end is -0.75");
}

/// The outputs of every sampled point lie in the enclosure: support along random directions.
void the_enclosure_contains_the_outputs(const std::string &b) {
    cora::Rng rng(3);
    const NeuralNetwork nn = randomNetwork(rng, {3, 6, 5, 2});
    const Zonotope X = Zonotope::generateRandom(3, 5, rng);
    const Zonotope Y = nn.evaluate(X);
    int violations = 0;
    for (const std::string type : {"standard", "extreme"}) {
        const std::vector<double> y = nn.evaluate(X.randPoint(200, rng, type)).data();  // rows 0, 1
        for (int k = 0; k < 10; ++k) {
            const std::vector<double> d = test::random_direction(rng, 2);
            const double bound = test::support(Y, d);
            for (int j = 0; j < 200; ++j)
                violations += d[0] * y[j] + d[1] * y[200 + j] > bound + 1e-9;
        }
    }
    check(violations == 0, b + ": " + std::to_string(violations) + " outputs left the enclosure");
}

void wrong_arguments_are_described(const std::string &b) {
    const Tensor W = Tensor({{1, 2}, {3, 4}}), bias = Tensor({0.0, 0.0});
    check(test::throws([&] { NeuralNetwork(std::vector<NeuralNetwork::Layer>{}); }), b + ": no layers");
    check(test::throws([&] { NeuralNetwork({{W, Tensor({0.0, 0.0, 0.0})}}); }),
          b + ": a bias of another size");
    check(test::throws([&] { NeuralNetwork({{W, bias}, {Tensor({{1, 2, 3}}), Tensor({0.0})}}); }),
          b + ": layers that do not fit");
    const NeuralNetwork nn({{W, bias}});
    check(test::throws([&] { nn.evaluate(Tensor({1.0, 2.0, 3.0})); }),
          b + ": a point of another dimension");
    check(test::throws([&] { nn.evaluate(Zonotope(Tensor({1.0}), Tensor({{1.0}}))); }),
          b + ": a set of another dimension");
}

} // namespace

int main() {
    test::for_each_backend([](const std::string &b) {
        points_follow_the_layers(b);
        an_affine_network_maps_the_set_exactly(b);
        stable_neurons_are_exact(b);
        a_crossing_neuron_gets_the_tightest_parallelogram(b);
        the_enclosure_contains_the_outputs(b);
        wrong_arguments_are_described(b);
    });
    return test::finish("neuralNetwork evaluate");
}
