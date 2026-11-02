// generate_random - a random interval, as CORA's interval.generateRandom
//
// Syntax:   Interval I = Interval::generate_random(n, rng);
// Inputs:   n - dimension; rng - random numbers (its seed makes the result repeatable)
// Outputs:  I - box with center ~ U[-2, 2]^n and radius R/2*u, R ~ U[0, 10], u ~ U[0, 1]
// See also: Zonotope::generate_random
//
// CORA: contSet/@interval/generateRandom.m

#include "cora/cont_set/interval/interval.h"

#include "cora/global/rng.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// `count` numbers uniform in [low, high].
std::vector<double> aux_uniform(Rng &rng, int64_t count, double low, double high) {
    std::vector<double> values(count);
    rng.uniform(values.data(), values.size(), low, high);
    return values;
}

} // namespace


// ===========================================  MAIN  =========================================== //

Interval Interval::generate_random(int64_t n, Rng &rng) {
    detail::Entry entry;
    CORA_CHECK(n >= 1, Error::wrong_value, [&] {
        return "generate_random: needs a dimension n >= 1, got " + std::to_string(n);
    });
    const std::vector<double> c = aux_uniform(rng, n, -2.0, 2.0);
    const std::vector<double> R = aux_uniform(rng, n, 0.0, 10.0);
    const std::vector<double> u = aux_uniform(rng, n, 0.0, 1.0);

    std::vector<double> inf(n), sup(n);
    for (int64_t i = 0; i < n; ++i) {
        const double radius = 0.5 * R[i] * u[i];
        inf[i] = c[i] - radius;
        sup[i] = c[i] + radius;
    }
    return {Tensor::from_data(inf, {n, 1}), Tensor::from_data(sup, {n, 1})};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
