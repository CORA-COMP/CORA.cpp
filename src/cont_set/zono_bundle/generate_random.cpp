// generate_random - a random zonotope bundle, as CORA's zonoBundle.generateRandom
//
// Random zonotopes about one random center p, each moved by a random part of the vector from p to a
// support point of a random direction: p stays inside every member.
//
// Syntax:   ZonoBundle bundle = ZonoBundle::generate_random(rng); ZonoBundle::generate_random(rng,
// n, k); Inputs:   rng - random numbers
//           dimension - n (default 1..10)
//           nr_zonotopes - k (default 2..5)
// Outputs:  bundle - k zonotopes of dimension n with a common point
// See also: Zonotope::generate_random
//
// CORA: contSet/@zonoBundle/generateRandom.m
// CORA authors: Mark Wetzlinger, Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/cont_set/private/priv_random_dimension.h"
#include "cora/cont_set/zono_bundle/zono_bundle.h"
#include "cora/global/rng.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

CORA_FROM_NUMBERS ZonoBundle ZonoBundle::generate_random(Rng &rng, std::optional<int64_t> dimension,
                                                         std::optional<int64_t> nr_zonotopes) {
    detail::Entry entry;
    CORA_CHECK(!dimension || *dimension >= 1, Error::wrong_value, [&] {
        return "generate_random: the dimension must be at least 1, got " +
               std::to_string(*dimension);
    });
    CORA_CHECK(!nr_zonotopes || *nr_zonotopes >= 1, Error::wrong_value, [&] {
        return "generate_random: nr_zonotopes must be at least 1, got " +
               std::to_string(*nr_zonotopes);
    });
    const int64_t n = dimension ? *dimension : priv_random_dimension(rng);
    // CORA: 1 + randi(4) zonotopes
    double draw = 0.0;
    rng.uniform(&draw, 1, 0.0, 4.0);
    const int64_t k = nr_zonotopes ? *nr_zonotopes : 2 + int64_t(draw);
    std::vector<double> p(static_cast<std::size_t>(n));
    rng.normal(p.data(), p.size(), 1.0);
    std::vector<Zonotope> members;
    for (int64_t i = 0; i < k; ++i) {
        const Tensor center = Tensor::from_data(p, {n, 1});
        const Zonotope Z = Zonotope::generate_random(rng, n, center);
        // a vertex of Z in a random direction, and a random part of the way to it
        std::vector<double> d(static_cast<std::size_t>(n)), s(1);
        rng.uniform(d.data(), d.size(), 0.0, 1.0);
        rng.uniform(s.data(), 1, 0.0, 1.0);
        const Tensor direction = Tensor::like(center, d, {n, 1});
        const Tensor v =
            Z.center() +
            Z.generators().matmul(Z.generators().matrix_transpose().matmul(direction).sign());
        members.emplace_back(Z.center() + (v - center) * s[0], Z.generators());
    }
    return ZonoBundle(std::move(members));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
