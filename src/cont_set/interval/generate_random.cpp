// generate_random - a random interval, as CORA's interval.generateRandom
//
// Syntax:   Interval I = Interval::generate_random(rng);
//           I = Interval::generate_random(rng, dimension, center, max_radius);
// Inputs:   rng - random numbers (its seed makes the result repeatable)
//           dimension - default the dimension of center or max_radius, else random in 1..10
//           center - a vector, default U[-2, 2] per entry
//           max_radius - radii >= 0, or one number for all entries, default U[0, 10] per entry
// Outputs:  I - box with the center and radius max_radius/2*u, u ~ U[0, 1] per entry
// See also: Zonotope::generate_random
//
// CORA: contSet/@interval/generateRandom.m
// CORA authors: Mark Wetzlinger, Tobias Ladner
// Authors:       Tobias Ladner
// Written:       02-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/interval/interval.h"

#include "cora/global/rng.h"

#include "cont_set/cont_set/private/priv_random_dimension.h"

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

CORA_FROM_NUMBERS Interval Interval::generate_random(Rng &rng, std::optional<int64_t> dimension,
                                                     const std::optional<Tensor> &center,
                                                     const std::optional<Tensor> &max_radius) {
    detail::Entry entry;
    CORA_CHECK(!dimension || *dimension >= 1, Error::wrong_value, [&] {
        return "generate_random: the dimension must be at least 1, got " +
               std::to_string(*dimension);
    });
    CORA_CHECK(!max_radius || max_radius->min().item() >= 0, Error::wrong_value,
               [] { return std::string("generate_random: max_radius must be nonnegative"); });
    // the dimension: given, else that of the center or the radii, else random
    int64_t n = 0;
    if (dimension)
        n = *dimension;
    else if (center)
        n = center->size(-2);
    else if (max_radius && max_radius->numel() > 1)
        n = max_radius->size(-2);
    else
        n = priv_random_dimension(rng);
    // what is drawn lives where the given center or radii do, else on the current backend
    const Tensor *given = center ? &*center : max_radius ? &*max_radius : nullptr;
    const auto column = [&](double low, double high) {
        const std::vector<double> values = aux_uniform(rng, n, low, high);
        return given ? Tensor::like(*given, values, {n, 1}) : Tensor::from_data(values, {n, 1});
    };
    const Tensor c = center ? *center : column(-2.0, 2.0);
    Tensor r = max_radius ? *max_radius : column(0.0, 10.0);
    if (r.numel() == 1) r = r.reshape({1, 1}).tile({n, 1});
    const Shape vector{n, 1};
    CORA_CHECK(c.shape() == vector && r.shape() == vector, Error::wrong_value, [&] {
        return "generate_random: the center and max_radius must be vectors of the "
               "dimension " +
               std::to_string(n);
    });
    // the radius, symmetric to keep the center
    const Tensor radius = (r * 0.5).multiply(column(0.0, 1.0));
    return {c - radius, c + radius};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
