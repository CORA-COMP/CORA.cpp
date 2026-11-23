// test_neural_network_reduce_network - CORA's computeReducedNetwork
//
// Neurons whose output range settles within a bucket's tolerance are merged away and their range
// is added to the next layer as an interval; the output set contains the outputs of the original
// network, for static and dynamic buckets, a target rate, and input compression.
//
// CORA: nn/@neuralNetwork/computeReducedNetwork.m
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

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Hidden neuron i has the output near 0.1 i (ReLU of 0.001 x_0 + 0.1 i), then one output.
NeuralNetwork aux_network(Activation a, int64_t inputs = 2) {
    std::vector<double> W, b, V;
    for (int64_t i = 0; i < 10; ++i) {
        for (int64_t j = 0; j < inputs; ++j) W.push_back(j == 0 ? 0.001 : 0.0);
        b.push_back(0.1 * double(i));
        V.push_back(1.0 + 0.1 * double(i));
    }
    return NeuralNetwork({{Tensor::from_data(W, {10, inputs}), Tensor::from_data(b, {10})},
                          {Tensor::from_data(V, {1, 10}), Tensor({0.5})}},
                         {a, Activation::identity});
}

/// Whether the zonotope bounds the outputs of the network at points of the input zonotope in
/// random directions.
bool aux_contains(const NeuralNetwork &nn, const Zonotope &X, const Zonotope &Y) {
    // 50 points of each kind, 10 directions
    cora::Rng rng(5);
    const int64_t m = nn.output_dim();
    // the support of the set in each direction bounds the outputs
    for (const std::string type : {"standard", "extreme"}) {
        const std::vector<double> y = nn.evaluate(X.rand_point(50, rng, type)).data();
        for (int k = 0; k < 10; ++k) {
            const std::vector<double> d = test::random_direction(rng, int(m));
            const double bound = test::support(Y, d);
            // the point's value along the direction
            for (int j = 0; j < 50; ++j) {
                double v = 0;
                for (int64_t i = 0; i < m; ++i) v += d[std::size_t(i)] * y[std::size_t(j * m + i)];
                if (v > bound + 1e-9) return false;
            }
        }
    }
    return true;
}

int64_t aux_count(const Tensor &mask) {
    int64_t n = 0;
    for (const double v : mask.data()) n += v != 0.0;
    return n;
}

void reducing(const std::string &b) {
    const NeuralNetwork nn = aux_network(Activation::relu);
    const Zonotope X(Tensor({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 0.5}}));
    // static buckets {0}: the neurons with range below the tolerance (0.01) merge: 0.1 i + 0.001
    ReduceOptions opt;
    opt.bucket_tol = {0.15};
    const ReducedNetwork R = nn.compute_reduced_network(X, opt);
    check(aux_count(R.merged[0]) == 2 && R.net.layers()[0].W.shape()[0] == 8 &&
              R.net.layers()[1].W.shape()[1] == 8 && R.offset_center[1].defined() &&
              !R.offset_center[0].defined() && !R.merged[1].defined(),
          b + ": two neurons merged, the next layer keeps their range");
    check(aux_contains(nn, X, R.output), b + ": the output contains the original outputs");
    check(R.reduction_rate.size() == 3 && test::close(R.reduction_rate[1], 0.8, 1e-12),
          b + ": the rate of neurons that stay");
    // a target rate of 0.5 is found by bisection (merging the five lowest neurons)
    ReduceOptions rated;
    rated.reduction_rate = {0.5};
    const ReducedNetwork T = nn.compute_reduced_network(X, rated);
    check(aux_count(T.merged[0]) == 5 && aux_contains(nn, X, T.output),
          b + ": a target rate reached by the bisection");
    // a rate of 1 asks for no reduction; a rate per layer
    rated.reduction_rate = {1.0};
    const ReducedNetwork N = nn.compute_reduced_network(X, rated);
    check(!N.merged[0].defined() && N.net.layers()[0].W.shape()[0] == 10 &&
              aux_contains(nn, X, N.output),
          b + ": a rate of 1 keeps the network");
    // dynamic buckets (the neurons' own centers, shuffled) and a tolerance per layer
    ReduceOptions dyn;
    dyn.dynamic_buckets = true;
    dyn.bucket_tol = {0.11, 0.11};
    dyn.seed = 3;
    const ReducedNetwork D = nn.compute_reduced_network(X, dyn);
    check(aux_count(D.merged[0]) > 0 && aux_contains(nn, X, D.output),
          b + ": dynamic buckets merge neurons and stay sound");
    // the sigmoid's buckets are 0 and 1
    const NeuralNetwork sig = aux_network(Activation::sigmoid);
    ReduceOptions so;
    so.bucket_tol = {0.6};
    check(aux_contains(sig, X, sig.compute_reduced_network(X, so).output),
          b + ": a sigmoid network");
}

void compression(const std::string &b) {
    const NeuralNetwork nn = aux_network(Activation::relu, 3);
    const Zonotope X(Tensor({0.1, 0.1, 2.0}),
                     Tensor({{0.05, 0.0, 0.0}, {0.0, 0.05, 0.0}, {0.0, 0.0, 0.1}}));
    ReduceOptions opt;
    opt.input_compression = true;
    opt.bucket_tol = {0.2};
    const ReducedNetwork R = nn.compute_reduced_network(X, opt);
    // the first layer maps the kept input back (then a ReLU, which merges the small ones again)
    check(R.net.num_layers() == 3 && R.net.layers()[0].W.shape()[1] == 1 &&
              R.net.layers()[0].W.shape()[0] <= 2 && aux_contains(nn, X, R.output),
          b + ": the input is compressed and the output is enclosed");
    // a target rate on top of the compression
    ReduceOptions rated = opt;
    rated.reduction_rate = {0.5};
    check(aux_contains(nn, X, nn.compute_reduced_network(X, rated).output),
          b + ": a target rate with compression");
    // a negative input cannot be compressed
    const Zonotope neg(Tensor({0.0, 0.0, 0.0}),
                       Tensor({{1.0, 0.0, 0.0}, {0.0, 1.0, 0.0}, {0.0, 0.0, 1.0}}));
    check(test::raises([&] { nn.compute_reduced_network(neg, opt); }, Error::special_error),
          b + ": a negative input");
}

void mistakes(const std::string &b) {
    const NeuralNetwork nn = aux_network(Activation::relu);
    const Zonotope X(Tensor({0.0, 0.0}), Tensor({{1.0, 0.0}, {0.0, 0.5}}));
    check(
        test::raises([&] { nn.compute_reduced_network(Zonotope(Tensor({0.0}), Tensor({{1.0}}))); },
                     Error::dimension_mismatch),
        b + ": a set of another dimension");
    // a tolerance per layer of the wrong size, negative searches, and the rest
    ReduceOptions bad;
    bad.bucket_tol = {0.1, 0.2, 0.3};
    check(test::raises([&] { nn.compute_reduced_network(X, bad); }, Error::wrong_value),
          b + ": a tolerance per layer of the wrong size");
    bad = ReduceOptions{};
    bad.min_searches = -1;
    check(test::raises([&] { nn.compute_reduced_network(X, bad); }, Error::wrong_value),
          b + ": negative searches");
    check(test::raises([&] { aux_network(Activation::square).compute_reduced_network(X); },
                       Error::not_supported),
          b + ": an activation without buckets");
    bad = ReduceOptions{};
    bad.reduction_rate = {-1.0};
    check(test::raises([&] { nn.compute_reduced_network(X, bad); }, Error::not_converged),
          b + ": a rate nothing reaches");
    {
        // a valid call without checks
        ChecksScope off(false);
        check(nn.compute_reduced_network(X).net.num_layers() == 2,
              b + ": a valid call without checks");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) {
        reducing(b);
        compression(b);
        mistakes(b);
    });
    return test::finish("neuralNetwork reduce network");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
