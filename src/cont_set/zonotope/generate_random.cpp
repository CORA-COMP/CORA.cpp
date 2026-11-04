// generate_random - a random zonotope, as CORA's zonotope.generateRandom
//
// Syntax:   Zonotope Z = Zonotope::generate_random(rng);
//           Z = Zonotope::generate_random(rng, dimension, center, nr_generators, distribution);
// Inputs:   rng - random numbers (seedable);  dimension - default the dimension of center, else
//           random in 1..10;  center - a vector, default 10 N(0, 1) per entry;  nr_generators -
//           default 2 dimension;  distribution - of the generator lengths: "uniform" (default,
//           U[0, 1]), "exp" (exponential, rate 1) or "gamma" (shape 2, scale 1)
// Outputs:  Z - the center and generators of uniformly random direction
// See also: Interval::generate_random
//
// CORA: contSet/@zonotope/generateRandom.m

#include "cora/cont_set/zonotope/zonotope.h"

#include "cora/global/rng.h"

#include "cont_set/cont_set/private/priv_random_dimension.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// n x m generators (row-major): Gaussian columns scaled to a length from the distribution.
std::vector<double> aux_generators(int64_t n, int64_t m, const std::string &distribution,
                                   Rng &rng) {
    std::vector<double> G(n * m), length(m), more(m);
    rng.normal(G.data(), G.size(), 1.0);
    rng.uniform(length.data(), length.size(), 0.0, 1.0);
    if (distribution != "uniform") {
        // exponential(1) by inversion; gamma(2, 1) is the sum of two of them
        for (double &l : length) l = -std::log1p(-l);
        if (distribution == "gamma") {
            rng.uniform(more.data(), more.size(), 0.0, 1.0);
            for (int64_t j = 0; j < m; ++j) length[j] -= std::log1p(-more[j]);
        }
    }
    for (int64_t j = 0; j < m; ++j) {
        double norm = 0.0;
        for (int64_t i = 0; i < n; ++i) norm += G[i * m + j] * G[i * m + j];
        norm = std::sqrt(norm);
        for (int64_t i = 0; i < n; ++i) G[i * m + j] *= length[j] / norm;
    }
    return G;
}

} // namespace


// ===========================================  MAIN  =========================================== //

CORA_FROM_NUMBERS Zonotope Zonotope::generate_random(Rng &rng, std::optional<int64_t> dimension,
                                                     const std::optional<Tensor> &center,
                                                     std::optional<int64_t> nr_generators,
                                                     const std::string &distribution) {
    detail::Entry entry;
    CORA_CHECK(!dimension || *dimension >= 1, Error::wrong_value, [&] {
        return "generate_random: the dimension must be at least 1, got " +
               std::to_string(*dimension);
    });
    CORA_CHECK(!nr_generators || *nr_generators >= 0, Error::wrong_value, [&] {
        return "generate_random: nr_generators must be nonnegative, got " +
               std::to_string(*nr_generators);
    });
    CORA_CHECK(distribution == "uniform" || distribution == "exp" || distribution == "gamma",
               Error::wrong_value, [&] {
                   return "generate_random: the distribution must be 'uniform', 'exp' or "
                          "'gamma', got '" +
                          distribution + "'";
               });
    // the dimension: given, else that of the center, else random
    int64_t n = 0;
    if (dimension)
        n = *dimension;
    else if (center)
        n = center->size(-2);
    else
        n = priv_random_dimension(rng);
    const Shape vector{n, 1};
    CORA_CHECK(!center || center->shape() == vector, Error::wrong_value, [&] {
        return "generate_random: the center must be a vector of the dimension " + std::to_string(n);
    });
    const int64_t m = nr_generators ? *nr_generators : 2 * n;
    std::vector<double> c(n);
    if (!center) rng.normal(c.data(), c.size(), 10.0);
    // the generators live where the given center does, else on the current backend
    const std::vector<double> G = aux_generators(n, m, distribution, rng);
    if (center) return {*center, Tensor::like(*center, G, {n, m})};
    return {Tensor::from_data(c, {n, 1}), Tensor::from_data(G, {n, m})};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
