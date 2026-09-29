// generateRandom - a random interval, as CORA's interval.generateRandom
//
// Syntax:   Interval I = Interval::generateRandom(n, rng);
// Inputs:   n - dimension; rng - random numbers (its seed makes the result repeatable)
// Outputs:  I - box with center ~ U[-2, 2]^n and radius R/2*u, R ~ U[0, 10], u ~ U[0, 1]
// See also: Zonotope::generateRandom

#include "contSet/interval/interval.h"

#include "global/rng.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::ct {

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

Interval Interval::generateRandom(int64_t n, Rng &rng) {
    const std::vector<double> c = aux_uniform(rng, n, -2.0, 2.0);
    const std::vector<double> R = aux_uniform(rng, n, 0.0, 10.0);
    const std::vector<double> u = aux_uniform(rng, n, 0.0, 1.0);

    std::vector<double> inf(n), sup(n);
    for (int64_t i = 0; i < n; ++i) {
        const double radius = 0.5 * R[i] * u[i];
        inf[i] = c[i] - radius;
        sup[i] = c[i] + radius;
    }
    return {Tensor::fromData(inf, {n, 1}), Tensor::fromData(sup, {n, 1})};
}

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
