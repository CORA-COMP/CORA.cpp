// test_neural_network_random_network - random networks, drawn weights, casts and copies
//
// A random network has the sizes asked for, entries in [-1, 1) and the activation on every layer;
// the sizes not given are drawn in CORA's ranges, and one seed gives one network. initWeights draws
// the dense layers' weights by the bounds of glorot (uniform in +-1 / sqrt(inputs)) or the spread
// of shi (standard deviation sqrt(2 pi) / inputs), the biases 0, only in the layers asked for, and
// leaves convolutions alone. A cast network computes the same up to the dtype, a copy is
// independent of the changes made to the original.
//
// PROPERTY: nn/@neuralNetwork/generateRandom.m
// PROPERTY: nn/@neuralNetwork/initWeights.m
// PROPERTY: nn/@neuralNetwork/castWeights.m
// PROPERTY: nn/@neuralNetwork/copyNeuralNetwork.m
// CORA: nn/@neuralNetwork/generateRandom.m
// CORA: nn/@neuralNetwork/initWeights.m
// CORA: nn/@neuralNetwork/castWeights.m
// CORA: nn/@neuralNetwork/copyNeuralNetwork.m
// CORA authors: Tobias Ladner, Lukas Koller
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
using Layer = NeuralNetwork::Layer;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The largest absolute entry of a tensor.
double aux_max(const Tensor &t) { return t.abs().max().item(); }

/// The sizes, the entries and the activations of random networks.
void random_networks(const std::string &b) {
    RandomOptions o;
    o.inputs = 3;
    o.outputs = 2;
    o.layers = 3;
    o.hidden = 7;
    o.activation = Activation::tanh;
    const NeuralNetwork nn = NeuralNetwork::random(o);
    bool in_range = true, acts = true;
    for (std::size_t k = 0; k < 3; ++k) {
        in_range = in_range && aux_max(nn.layers()[k].W) < 1.0 && aux_max(nn.layers()[k].b) < 1.0;
        acts = acts && nn.activations()[k] == Activation::tanh;
    }
    check(nn.num_layers() == 3 && nn.input_dim() == 3 && nn.output_dim() == 2 &&
              nn.layers()[0].outputs() == 7 && nn.layers()[1].outputs() == 7,
          b + ": the sizes asked for");
    check(in_range && acts, b + ": entries in [-1, 1), the activation of every layer");
    // one seed, one network; another seed, another
    check(test::close(NeuralNetwork::random(o).layers()[0].W, nn.layers()[0].W, 0.0),
          b + ": one seed gives one network");
    o.seed = 2;
    check(aux_max(NeuralNetwork::random(o).layers()[0].W - nn.layers()[0].W) > 0,
          b + ": another seed another network");
    // sizes drawn in CORA's ranges
    bool ranges = true;
    for (uint64_t seed = 1; seed <= 30; ++seed) {
        const NeuralNetwork r = NeuralNetwork::random({.seed = seed});
        ranges = ranges && r.input_dim() >= 1 && r.input_dim() <= 5 && r.output_dim() >= 1 &&
                 r.output_dim() <= 5 && r.num_layers() >= 1 && r.num_layers() <= 5 &&
                 r.activations().front() == Activation::sigmoid;
        if (r.num_layers() > 1)
            ranges = ranges && r.layers()[0].outputs() >= 2 && r.layers()[0].outputs() <= 20;
    }
    check(ranges, b + ": the sizes not given are drawn in CORA's ranges");
    check(test::raises([] { NeuralNetwork::random({.inputs = -1}); }, Error::wrong_value),
          b + ": a negative size is refused");
}

/// The weights drawn again by each method, in the layers asked for.
void init_weights(const std::string &b) {
    const NeuralNetwork nn = NeuralNetwork::random({.inputs = 40, .outputs = 300, .layers = 1});
    const NeuralNetwork glorot = nn.init_weights("glorot", 5);
    const Tensor &W = glorot.layers()[0].W;
    check(aux_max(W) <= 1.0 / std::sqrt(40.0) && aux_max(W) > 0.9 / std::sqrt(40.0) &&
              aux_max(glorot.layers()[0].b) == 0.0,
          b + ": glorot is uniform in +-1 / sqrt(inputs), the biases 0");
    // shi: the standard deviation of 12000 entries
    const Tensor S = nn.init_weights("shi", 5).layers()[0].W;
    const double sigma = std::sqrt(2.0 * 3.14159265358979323846) / 40.0,
                 var = S.square().sum().item() / double(S.numel());
    check(std::abs(std::sqrt(var) / sigma - 1.0) < 0.05, b + ": shi has the standard deviation");
    check(test::close(nn.init_weights("glorot", 5).layers()[0].W, W, 0.0) &&
              aux_max(nn.init_weights("glorot", 6).layers()[0].W - W) > 0,
          b + ": the seed fixes the draw");
    // only the layers asked for, a convolution stays
    NeuralNetwork::Conv2d spec;
    const Layer conv =
        Layer::conv2d(Tensor::ones({1, 1, 2, 2}), Tensor::zeros({4}), {1, 3, 3}, spec);
    const Layer dense{Tensor::ones({2, 4}), Tensor({1.0, 1.0})};
    const NeuralNetwork mixed({conv, dense}, {Activation::relu, Activation::identity});
    const NeuralNetwork both = mixed.init_weights("glorot", 1);
    check(test::close(both.layers()[0].dense(), conv.dense(), 0.0) &&
              aux_max(both.layers()[1].W - dense.W) > 0,
          b + ": a convolution keeps its kernel, a dense layer is drawn");
    const NeuralNetwork first = nn.init_weights("glorot", 1, {0});
    check(aux_max(first.layers()[0].W - nn.layers()[0].W) > 0 &&
              test::close(mixed.init_weights("glorot", 1, {0}).layers()[1].W, dense.W, 0.0),
          b + ": only the layers asked for");
    check(test::raises([&] { nn.init_weights("xavier"); }, Error::wrong_value) &&
              test::raises([&] { nn.init_weights("glorot", 1, {4}); }, Error::wrong_value),
          b + ": an unknown method or a layer out of range is refused");
}

/// The cast and the copy.
void cast_and_copy(const std::string &b) {
    const NeuralNetwork nn = NeuralNetwork::random({.inputs = 3, .outputs = 2, .layers = 2});
    // a point drawn from a fixed seed
    Rng rng(4);
    std::vector<double> point(3);
    rng.uniform(point.data(), point.size(), -1.0, 1.0);
    const Tensor x = Tensor::from_data(point, {1, 3});
    const NeuralNetwork single = nn.to(DType::float32);
    check(single.layers()[0].W.dtype() == DType::float32 &&
              nn.layers()[0].W.dtype() == DType::float64,
          b + ": the weights are cast");
    check(test::close(single.evaluate(x.to(DType::float32)), nn.evaluate(x).data(), 1e-5),
          b + ": the cast network computes the same up to the dtype");
    // a copy drawn again changes, the original keeps its weights
    const Tensor before = nn.evaluate(x);
    NeuralNetwork copy = nn;
    copy = copy.init_weights("shi", 3);
    check(test::close(nn.evaluate(x), before, 0.0) && aux_max(copy.evaluate(x) - before) > 0,
          b + ": a copy is independent of the changes");
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        random_networks(b);
        init_weights(b);
        cast_and_copy(b);
    });
    // a valid call with the checks off
    {
        ChecksScope off(false);
        check(NeuralNetwork::random({.inputs = 2, .outputs = 1, .layers = 1}).num_layers() == 1,
              "checks off: a valid call");
    }
    return test::finish("neuralNetwork random network");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
