// generate_random - a random capsule, as CORA's capsule.generateRandom
//
// The dimension is given, else the length of the center, else uniform in 1..10; the center is
// 5 N(0, 1)^n and the radius U[0, 5] unless given; the generator is U[0, 1]^n.
//
// Syntax:   C = Capsule::generate_random(rng);   C = Capsule::generate_random(rng, n, center,
// radius); Inputs:   rng - the generator of the random numbers
//           n, center, radius - optional, as CORA's name-value pairs Dimension, Center, Radius
// Outputs:  C - the capsule
// See also: Zonotope::generate_random
//
// CORA: contSet/@capsule/generateRandom.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/capsule/capsule.h"
#include "cora/global/rng.h"

#include "cont_set/cont_set/private/priv_random_dimension.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

CORA_FROM_NUMBERS Capsule Capsule::generate_random(Rng &rng, std::optional<int64_t> dimension,
                                                   const std::optional<Tensor> &center,
                                                   std::optional<double> radius) {
    detail::Entry entry;
    CORA_CHECK(!dimension || *dimension >= 1, Error::wrong_value, [&] {
        return "generate_random: the dimension must be at least 1, got " +
               std::to_string(*dimension);
    });
    CORA_CHECK(
        !center || (center->ndim() >= 2 && center->shape().back() == 1), Error::wrong_value,
        [] { return std::string("generate_random: the center must be a column (..., n, 1)"); });
    CORA_CHECK(!(dimension && center) || center->size(-2) == *dimension, Error::dimension_mismatch,
               [] { return std::string("generate_random: the center and the dimension differ"); });
    const int64_t n = dimension ? *dimension
                      : center  ? center->size(-2)
                                : priv_random_dimension(rng);
    CORA_CHECK(!radius || *radius >= 0.0, Error::wrong_value, [&] {
        return "generate_random: the radius must be nonnegative, got " + format_number(*radius);
    });
    // c = 5 randn(n, 1), r = 5 rand, g = rand(n, 1), in the order of CORA
    std::vector<double> c(static_cast<std::size_t>(n)), g(c.size()), r(1);
    if (!center) rng.normal(c.data(), c.size(), 5.0);
    if (!radius) rng.uniform(r.data(), 1, 0.0, 5.0);
    rng.uniform(g.data(), g.size(), 0.0, 1.0);
    const Tensor generator = Tensor::from_data(g, {n, 1});
    return {center ? *center : Tensor::from_data(c, {n, 1}), generator, radius ? *radius : r[0]};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
