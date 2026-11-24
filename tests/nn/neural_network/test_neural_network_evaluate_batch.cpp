// test_neural_network_evaluate_batch - a batch of zonotopes through a network
//
// The enclosure of a batch holds sampled outputs through a softmax and a max pool, with the lines
// of "bounds" and "singh" and with errors in the radius of the interval center; a linear layer is
// exact.
//
// CORA: nn/@neuralNetwork/evaluateZonotopeBatch.m
// CORA authors: Tobias Ladner, Lukas Koller
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/nn/neural_network/neural_network.h"
#include "testing.h"

#include <random>

using namespace cora;
using test::check;

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The values of n random numbers in [-2, 2].
std::vector<double> aux_random(std::mt19937 &gen, int64_t n) {
    std::uniform_real_distribution<double> d(-2.0, 2.0);
    std::vector<double> x(static_cast<std::size_t>(n), 0.0);
    for (double &v : x) v = d(gen);
    return x;
}

/// The batch of boxes through the network holds sampled outputs.
int aux_violations(const NeuralNetwork &nn, const std::vector<double> &c,
                   const std::vector<double> &r, const ZonotopeBatchOptions &o, std::mt19937 &gen) {
    // the box as a batch of one: the center and a generator per radius
    const int64_t n = int64_t(c.size());
    std::vector<double> G(std::size_t(n * n), 0.0);
    for (int64_t i = 0; i < n; ++i) G[std::size_t(i * n + i)] = r[std::size_t(i)];
    const ZonotopeBatch Y = nn.evaluate_zonotope_batch(Tensor::from_data(c, {1, n}),
                                                       Tensor::from_data(G, {1, n, n}), o);
    Tensor radius = Y.G.abs().sum(1);
    if (Y.dr.defined()) radius = radius + Y.dr;
    const std::vector<double> lo = (Y.c - radius).data(), hi = (Y.c + radius).data();
    // the samples of the box against the enclosure
    std::uniform_real_distribution<double> d(-1.0, 1.0);
    // the violations of the lower and upper end of every output
    int bad = 0;
    for (int s = 0; s < 200; ++s) {
        // a random point of the box and its output
        std::vector<double> x = c;
        for (int64_t i = 0; i < n; ++i) x[std::size_t(i)] += r[std::size_t(i)] * d(gen);
        const std::vector<double> y = nn.evaluate(Tensor::from_data(x, {1, n})).data();
        for (std::size_t i = 0; i < y.size(); ++i)
            bad += y[i] < lo[i] - 1e-9 || y[i] > hi[i] + 1e-9;
    }
    return bad;
}

void batches(const std::string &b) {
    // the lines of bounds and singh, and one error generator per layer
    std::mt19937 gen(3);
    ZonotopeBatchOptions singh;
    singh.poly_method = "singh";
    ZonotopeBatchOptions few;
    few.num_approx_err = 1;
    for (const ZonotopeBatchOptions &o : {ZonotopeBatchOptions{}, singh, few}) {
        check(aux_violations(NeuralNetwork(NeuralNetwork::softmax_layer(3)), {0.5, -0.5, 1.0},
                             {0.2, 0.3, 0.1}, o, gen) == 0,
              b + ": a box through a softmax (" + o.poly_method + ")");
        check(aux_violations(NeuralNetwork(NeuralNetwork::max_pool2d_layer({2, 2}, {1, 2, 4})),
                             aux_random(gen, 8), std::vector<double>(8, 0.4), o, gen) == 0,
              b + ": a box through a max pool (" + o.poly_method + ")");
    }
    // a linear network is exact: c' = W c + b, G' = G W^T, no new generators
    const NeuralNetwork lin({{Tensor({{1.0, 2.0}, {0.0, -1.0}}), Tensor({0.5, 0.0})}},
                            {Activation::identity});
    const ZonotopeBatch Y = lin.evaluate_zonotope_batch(
        Tensor({{1.0, 1.0}}), Tensor::from_data({1.0, 0.0, 0.0, 1.0, 2.0, 2.0}, {1, 3, 2}));
    check(test::close(Y.c, std::vector<double>{3.5, -1.0}) &&
              test::close(Y.G, std::vector<double>{1.0, 0.0, 2.0, -1.0, 6.0, -2.0}) &&
              !Y.dr.defined(),
          b + ": a linear layer is exact, with more generators than inputs");
    check(test::raises(
              [&] {
                  lin.evaluate_zonotope_batch(Tensor({{1.0}}), Tensor::from_data({1.0}, {1, 1, 1}));
              },
              Error::dimension_mismatch),
          b + ": a center of the wrong size");
    {
        // a valid call without checks
        ChecksScope off(false);
        check(lin.evaluate_zonotope_batch(Tensor({{1.0, 1.0}}),
                                          Tensor::from_data({1.0, 0.0}, {1, 1, 2}))
                      .c.numel() == 2,
              b + ": a valid call without checks");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { batches(b); });
    return test::finish("neuralNetwork evaluate batch");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
