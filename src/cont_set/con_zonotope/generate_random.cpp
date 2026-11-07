// generate_random - a random constrained zonotope, as CORA's conZonotope.generateRandom
//
// A random zonotope gives the center and generators. With constraints, the factors of a random
// point of the box satisfy them: A x0 = b for a random A in [-5, 5] and x0 in [-1, 1].
//
// Syntax:   ConZonotope cz = ConZonotope::generate_random(rng);   ConZonotope::generate_random(rng,
// n, c, m, p); Inputs:   rng - random numbers
//           dimension - n (default: that of center, else 1..10)
//           center - the center (n, 1) (default: 5 randn)
//           nr_generators - default 2 n
//           nr_constraints - default random from 0 to nr_generators - n
// Outputs:  cz - a constrained zonotope; no constraints if there are none
// See also: Zonotope::generate_random
//
// CORA: contSet/@conZonotope/generateRandom.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "cont_set/cont_set/private/priv_random_dimension.h"
#include "cora/global/rng.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

CORA_FROM_NUMBERS ConZonotope ConZonotope::generate_random(Rng &rng,
                                                           std::optional<int64_t> dimension,
                                                           const std::optional<Tensor> &center,
                                                           std::optional<int64_t> nr_generators,
                                                           std::optional<int64_t> nr_constraints) {
    detail::Entry entry;
    // the sizes are checked before anything is drawn
    CORA_CHECK(!dimension || *dimension >= 1, Error::wrong_value, [&] {
        return "generate_random: the dimension must be at least 1, got " +
               std::to_string(*dimension);
    });
    CORA_CHECK(!nr_generators || *nr_generators >= 0, Error::wrong_value, [&] {
        return "generate_random: nr_generators must be nonnegative, got " +
               std::to_string(*nr_generators);
    });
    CORA_CHECK(!nr_constraints || *nr_constraints >= 0, Error::wrong_value, [&] {
        return "generate_random: nr_constraints must be nonnegative, got " +
               std::to_string(*nr_constraints);
    });
    const int64_t n =
        dimension ? *dimension : (center ? center->size(-2) : priv_random_dimension(rng));
    const int64_t m = nr_generators ? *nr_generators : 2 * n;
    // the number of constraints: given, else one of 0 .. m - n (CORA's randi)
    int64_t p = 0;
    if (nr_constraints) {
        p = *nr_constraints;
    } else if (m > n) {
        double draw = 0.0;
        rng.uniform(&draw, 1, 0.0, double(m - n + 1));
        p = int64_t(draw);
    }
    const Zonotope Z = Zonotope::generate_random(rng, n, center, m);
    if (p == 0) return ConZonotope(Z);
    // a point of the box satisfies the constraints
    std::vector<double> x0(static_cast<std::size_t>(m)), A(static_cast<std::size_t>(p * m));
    rng.uniform(x0.data(), x0.size(), -1.0, 1.0);
    rng.uniform(A.data(), A.size(), -5.0, 5.0);
    const Tensor Ae = Tensor::like(Z.center(), A, {p, m});
    return {Z.center(), Z.generators(), Ae, Ae.matmul(Tensor::like(Z.center(), x0, {m, 1}))};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
