// generateRandom - a random zonotope, as CORA's zonotope.generateRandom
//
// Syntax:   Zonotope Z = Zonotope::generateRandom(n, m, rng);
// Inputs:   n - dimension; m - number of generators; rng - random numbers (seedable)
// Outputs:  Z - center 10*randn, m generators of uniformly random direction, length ~ U[0, 1]
// See also: Interval::generateRandom

#include "contSet/zonotope/zonotope.h"

#include "global/rng.h"

#include <cmath>

namespace cora::ct {

namespace {
std::vector<double> aux_generators(int64_t n, int64_t m, Rng &rng);
} // namespace

Zonotope Zonotope::generateRandom(int64_t n, int64_t m, Rng &rng) {
    std::vector<double> c(n);
    rng.normal(c.data(), c.size(), 10.0);
    return {Tensor::fromData(c, {n, 1}), Tensor::fromData(aux_generators(n, m, rng), {n, m})};
}

// --------------------------- auxiliary functions --------------------------------

namespace {

/// n x m generators (row-major): Gaussian columns scaled to a length ~ U[0, 1].
std::vector<double> aux_generators(int64_t n, int64_t m, Rng &rng) {
    std::vector<double> G(n * m), length(m);
    rng.normal(G.data(), G.size(), 1.0);
    rng.uniform(length.data(), length.size(), 0.0, 1.0);
    for (int64_t j = 0; j < m; ++j) {
        double norm = 0.0;
        for (int64_t i = 0; i < n; ++i) norm += G[i * m + j] * G[i * m + j];
        norm = std::sqrt(norm);
        for (int64_t i = 0; i < n; ++i) G[i * m + j] *= length[j] / norm;
    }
    return G;
}

} // namespace

} // namespace cora::ct
