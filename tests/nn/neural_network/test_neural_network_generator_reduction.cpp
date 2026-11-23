// test_neural_network_generator_reduction - CORA's generator reduction layer on a batch
//
// The generators with the longest L1 norms (CORA keeps the shortest) become a box, so that exactly
// max_gens remain; the reduced set contains the original in every direction, and a batch that
// fits stays as it is.
//
// CORA: nn/layers/linear/nnGeneratorReductionLayer.m
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

/// The support function of batch entry 0 along d: d c + sum |d g|.
double aux_support(const ZonotopeBatch &Z, const std::vector<double> &d) {
    const std::vector<double> c = Z.c.data(), G = Z.G.data();
    const std::size_t n = d.size(), q = std::size_t(Z.G.shape()[1]);
    double h = 0;
    for (std::size_t i = 0; i < n; ++i) h += d[i] * c[i];
    for (std::size_t k = 0; k < q; ++k) {
        double s = 0;
        for (std::size_t i = 0; i < n; ++i) s += d[i] * G[k * n + i];
        h += std::abs(s);
    }
    return h;
}

void reduction(const std::string &b) {
    // five generators of the L1 lengths 5, 1, 3, 2, 4 in two dimensions
    const Tensor G =
        Tensor::from_data({5.0, 0.0, 0.0, 1.0, 1.0, 2.0, 0.0, -2.0, -3.0, 1.0}, {1, 5, 2});
    const ZonotopeBatch Z{Tensor({{1.0, -1.0}}), G, Tensor()};
    // max_gens 4 keeps the two shortest (lengths 1 and 2, in that order), boxes the rest
    const ZonotopeBatch R = NeuralNetwork::reduce_generators(Z, 4);
    check(R.G.shape() == std::vector<int64_t>{1, 4, 2} &&
              test::close(R.G, {0.0, 1.0, 0.0, -2.0, 9.0, 0.0, 0.0, 3.0}, 1e-14) &&
              test::close(R.c, {1.0, -1.0}, 0.0) && !R.dr.defined(),
          b + ": the shortest stay, the longest become a box");
    // the reduced set contains the original in every direction
    for (const std::vector<double> &d :
         std::vector<std::vector<double>>{{1, 0}, {0, 1}, {1, 1}, {1, -2}, {-0.3, 0.9}})
        check(aux_support(R, d) >= aux_support(Z, d) - 1e-12, b + ": the support is not smaller");
    // all but the box: max_gens = n
    check(NeuralNetwork::reduce_generators(Z, 2).G.shape() == std::vector<int64_t>{1, 2, 2},
          b + ": only the box");
    // a batch that fits, and the radius of the interval center
    const ZonotopeBatch with_dr{Z.c, Z.G, Tensor({{0.5, 0.5}})};
    const ZonotopeBatch same = NeuralNetwork::reduce_generators(with_dr, 7);
    check(same.G.shape() == Z.G.shape() && same.dr.defined(), b + ": a batch that fits is kept");
    check(NeuralNetwork::reduce_generators(with_dr, 4).dr.defined(), b + ": dr is untouched");
    check(test::raises([&] { NeuralNetwork::reduce_generators(Z, 1); }, Error::wrong_value),
          b + ": fewer than n generators");
    check(
        test::raises([&] { NeuralNetwork::reduce_generators({Tensor({{1.0}}), Z.G, Tensor()}, 4); },
                     Error::dimension_mismatch),
        b + ": a center of another size");
    {
        // a valid call without checks
        ChecksScope off(false);
        check(NeuralNetwork::reduce_generators(Z, 3).G.shape()[1] == 3,
              b + ": a valid call without checks");
    }
}

} // namespace


// ===========================================  MAIN  =========================================== //

int main() {
    test::for_each_backend([](const std::string &b) { reduction(b); });
    return test::finish("neuralNetwork generator reduction");
}

// ---------------------------------------  END OF CODE  ---------------------------------------- //
