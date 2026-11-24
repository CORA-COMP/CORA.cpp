// test_neural_network_nonlinear_layers - CORA's exp, softmax and max-pool layers
//
// The blocks compute their layer's function on points (a softmax as 1 / sum_j exp(x_j - x_i), a
// maximum as b + relu(a - b)), and a batch of zonotopes through them stays enclosing.
//
// CORA: nn/layers/nonlinear/nnExpLayer.m
// CORA: nn/layers/nonlinear/nnSoftmaxLayer.m
// CORA: nn/layers/nonlinear/nnMaxPool2DLayer.m
// CORA: nn/layers/nonlinear/nnSoftPlusLayer.m
// CORA: nn/layers/nonlinear/nnRootLayer.m
// CORA: nn/layers/nonlinear/nnInvSqrtRootLayer.m
// CORA: nn/@neuralNetwork/evaluateZonotopeBatch.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/nn/neural_network/neural_network.h"
#include "testing.h"

#include <algorithm>
#include <cmath>
#include <random>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The maximum of every window of an image (C, H, W) in C order, for a pool and a stride.
std::vector<double> aux_max_pool(const std::vector<double> &x, const std::vector<int64_t> &in,
                                 std::array<int64_t, 2> pool, std::array<int64_t, 2> stride) {
    const int64_t ho = (in[1] - pool[0]) / stride[0] + 1, wo = (in[2] - pool[1]) / stride[1] + 1;
    // the quotients
    std::vector<double> y;
    for (int64_t c = 0; c < in[0]; ++c)
        for (int64_t i = 0; i < ho; ++i)
            for (int64_t j = 0; j < wo; ++j) {
                double best = -1e300;
                for (int64_t a = 0; a < pool[0]; ++a)
                    for (int64_t b = 0; b < pool[1]; ++b)
                        best =
                            std::max(best, x[std::size_t((c * in[1] + i * stride[0] + a) * in[2] +
                                                         j * stride[1] + b)]);
                y.push_back(best);
            }
    return y;
}

/// The values of n random numbers in [-2, 2].
std::vector<double> aux_random(std::mt19937 &gen, int64_t n) {
    std::uniform_real_distribution<double> d(-2.0, 2.0);
    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    for (double &v : x) v = d(gen);
    return x;
}

/// The softmax of a point, CORA's exp(x - max) / sum(exp(x - max)).
std::vector<double> aux_softmax(const std::vector<double> &x) {
    // shifted by the largest value, as CORA's
    const double top = *std::max_element(x.begin(), x.end());
    double total = 0;
    for (const double v : x) total += std::exp(v - top);
    std::vector<double> y;
    y.reserve(x.size());
    for (const double v : x) y.push_back(std::exp(v - top) / total);
    return y;
}

void exp_and_softmax(const std::string &b) {
    const NeuralNetwork e(NeuralNetwork::exp_layer(3));
    check(test::close(e.evaluate(Tensor({{0.0, 1.0, -2.0}})),
                      std::vector<double>{1.0, std::exp(1.0), std::exp(-2.0)}, 1e-14),
          b + ": exp layer");
    // softmax of one, two and five values against CORA's formula
    std::mt19937 gen(7);
    for (const int64_t n : {1, 2, 5}) {
        const NeuralNetwork s(NeuralNetwork::softmax_layer(n));
        check(s.num_layers() == 2 && s.output_dim() == n, b + ": the softmax is two layers");
        const std::vector<double> x = aux_random(gen, n);
        check(test::close(s.evaluate(Tensor::from_data(x, {1, n})), aux_softmax(x), 1e-12),
              b + ": softmax of " + std::to_string(n) + " values");
    }
    // a layer after a softmax: the block's sources are absolute
    NeuralNetwork::Block block = NeuralNetwork::softmax_layer(2, 0);
    std::vector<NeuralNetwork::Layer> layers{
        NeuralNetwork::Layer::linear(Tensor({{1.0, 0.0}, {0.0, 2.0}}), Tensor({0.0, 0.0}))};
    std::vector<Activation> acts{Activation::identity};
    layers.insert(layers.end(), block.layers.begin(), block.layers.end());
    acts.insert(acts.end(), block.activations.begin(), block.activations.end());
    // the block after a linear layer
    const NeuralNetwork chained(layers, acts);
    check(test::close(chained.evaluate(Tensor({{0.5, 0.25}})), aux_softmax({0.5, 0.5}), 1e-12),
          b + ": a softmax after a layer");
    check(test::raises([] { NeuralNetwork::softmax_layer(0); }, Error::wrong_input_in_constructor),
          b + ": a softmax of nothing");
}

void max_pool(const std::string &b) {
    // windows, strides and inputs: even and odd window sizes, channels, a 1 x 1 window
    std::mt19937 gen(11);
    struct Case {
        std::vector<int64_t> in;
        std::array<int64_t, 2> pool, stride;
    };
    for (const Case &k : std::vector<Case>{{{1, 4, 4}, {2, 2}, {0, 0}},
                                           {{2, 3, 4}, {2, 2}, {1, 2}},
                                           {{1, 3, 3}, {3, 3}, {0, 0}},
                                           {{2, 2, 5}, {1, 3}, {1, 1}},
                                           {{1, 3, 3}, {1, 1}, {0, 0}}}) {
        // the block against the window maxima of a random image
        const NeuralNetwork nn(NeuralNetwork::max_pool2d_layer(k.pool, k.in, k.stride));
        const std::array<int64_t, 2> st = k.stride[0] > 0 ? k.stride : k.pool;
        const std::vector<double> x = aux_random(gen, k.in[0] * k.in[1] * k.in[2]);
        const std::vector<double> want = aux_max_pool(x, k.in, k.pool, st);
        check(test::close(nn.evaluate(Tensor::from_data(x, {1, int64_t(x.size())})), want, 1e-12),
              b + ": max pool of " + std::to_string(k.pool[0]) + " x " + std::to_string(k.pool[1]) +
                  " windows");
    }
    check(test::raises([] { NeuralNetwork::max_pool2d_layer({4, 4}, {1, 3, 3}); },
                       Error::wrong_input_in_constructor),
          b + ": a pool larger than the image");
    {
        // valid calls without checks
        ChecksScope off(false);
        check(NeuralNetwork::max_pool2d_layer({2, 2}, {1, 2, 2}).layers.size() == 4 &&
                  NeuralNetwork::softmax_layer(2).layers.size() == 2,
              b + ": valid calls without checks");
    }
}

/// The points in the box [c - r, c + r] of the network lie in the zonotope's support in 20
/// directions.
int aux_violations(const NeuralNetwork &nn, const Zonotope &X, const Zonotope &Y, int out) {
    cora::Rng rng(9);
    int violations = 0;
    const std::vector<double> y =
        nn.evaluate(X.rand_point(100, rng, "extreme")).matrix_transpose().data();
    for (int k = 0; k < 10; ++k) {
        const std::vector<double> d = test::random_direction(rng, out);
        const double bound = test::support(Y, d);
        for (int j = 0; j < 100; ++j) {
            double v = 0;
            for (int i = 0; i < out; ++i)
                v += d[std::size_t(i)] * y[std::size_t(i) * 100 + std::size_t(j)];
            violations += v > bound + 1e-9;
        }
    }
    return violations;
}

/// The network with every layer as a matrix (sets take matrices only).
NeuralNetwork aux_dense(const NeuralNetwork &nn) {
    std::vector<NeuralNetwork::Layer> layers;
    for (const NeuralNetwork::Layer &L : nn.layers()) layers.push_back({L.dense(), L.b});
    return NeuralNetwork(layers, nn.activations());
}

void new_activations(const std::string &b) {
    const Tensor x({{0.5, 2.0}});
    // softplus with sharpness 1 and 2 (CORA: (1 / k) log(1 + e^(k x))), root, inverse square root
    check(test::close(NeuralNetwork(NeuralNetwork::softplus_layer(2)).evaluate(x),
                      {std::log(1 + std::exp(0.5)), std::log(1 + std::exp(2.0))}, 1e-14),
          b + ": softplus");
    const NeuralNetwork sharp(NeuralNetwork::softplus_layer(2, 2.0));
    check(sharp.num_layers() == 2 &&
              test::close(sharp.evaluate(x),
                          {0.5 * std::log(1 + std::exp(1.0)), 0.5 * std::log(1 + std::exp(4.0))},
                          1e-14),
          b + ": softplus of sharpness 2");
    check(test::close(NeuralNetwork(NeuralNetwork::root_layer(2)).evaluate(x),
                      {std::sqrt(0.5), std::sqrt(2.0)}, 1e-14) &&
              test::close(NeuralNetwork(NeuralNetwork::inv_sqrt_root_layer(2)).evaluate(x),
                          {std::pow(0.5, -0.5), std::pow(2.0, -0.5)}, 1e-14),
          b + ": root and inverse square root");
    check(test::raises([] { NeuralNetwork::softplus_layer(2, 0.0); },
                       Error::wrong_input_in_constructor),
          b + ": a sharpness of 0");
    // a set through each layer encloses the points, by both lines, and the batch and gradient
    const Zonotope X(Tensor({1.5, 2.5}), Tensor({{0.5, 0.0}, {0.0, 0.4}}));
    for (const NeuralNetwork::Block &block :
         {NeuralNetwork::softplus_layer(2, 2.0), NeuralNetwork::root_layer(2),
          NeuralNetwork::inv_sqrt_root_layer(2)}) {
        const NeuralNetwork nn(block);
        for (const PolyMethod m : {PolyMethod::singh, PolyMethod::regression})
            check(aux_violations(nn, X, aux_dense(nn).evaluate(X, {.poly_method = m}), 2) == 0,
                  b + ": the set through a layer of activation " +
                      std::to_string(int(nn.activations()[0])));
        const ZonotopeBatch out = nn.evaluate_zonotope_batch(
            Tensor({{1.5, 2.5}}), Tensor::from_data({0.5, 0.0, 0.0, 0.4}, {1, 2, 2}));
        const Zonotope Y(out.c.squeeze(0), out.G.squeeze(0));
        check(aux_violations(nn, X, Y, 2) == 0, b + ": the batch through the layer");
        // the gradient at a point against the Jacobian, and the box of the gradient
        const Tensor p({{1.5, 2.5}}), g({{1.0, 1.0}});
        check(test::close(nn.backprop(p, g),
                          aux_dense(nn).sensitivity(Tensor({1.5, 2.5})).sum(0).data(), 1e-12),
              b + ": the gradient is the Jacobian's");
        const auto [lo, hi] = nn.backprop_interval_batch(p, p, g, g);
        check(test::close(lo, nn.backprop(p, g), 1e-12) && test::close(hi, lo, 1e-12),
              b + ": the gradient bounds on a point");
    }
    // verify takes none of them
    check(
        test::raises([] { NeuralNetwork(NeuralNetwork::root_layer(2)).verify(NnProperty{}, 1.0); },
                     Error::not_supported),
        b + ": verify rejects the layers");
    // an order-3 polynomial zonotope takes their derivatives (Taylor), and stays enclosing
    for (const NeuralNetwork::Block &block :
         {NeuralNetwork::softplus_layer(2), NeuralNetwork::root_layer(2),
          NeuralNetwork::inv_sqrt_root_layer(2)}) {
        const NeuralNetwork nn(block);
        const PolyZonotope P(X);
        for (const PolyMethod m : {PolyMethod::taylor, PolyMethod::regression}) {
            const PolyZonotope Y =
                aux_dense(nn).evaluate(P, {.poly_method = m, .orders = {{3, 3}}});
            check(aux_violations(nn, X, Y.zonotope(), 2) == 0,
                  b + ": an order-3 polynomial zonotope through activation " +
                      std::to_string(int(nn.activations()[0])));
        }
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        exp_and_softmax(b);
        max_pool(b);
        new_activations(b);
    });
    return test::finish("neuralNetwork nonlinear layers");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
