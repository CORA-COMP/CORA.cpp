// rand_point - random points of a polynomial zonotope, as CORA's polyZonotope.randPoint_
//
// The factors are drawn uniformly in [-1, 1] (type "standard"): N points c + sum_j
// prod_i a_i^E(i,j) G(:,j) + GI b.
//
// Syntax:   p = pZ.rand_point(N, rng);
// Inputs:   N - the number of points
//           rng - the generator
// Outputs:  p - the points as columns (n, N)
// See also: zonotope
//
// CORA: contSet/@polyZonotope/randPoint_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "cora/global/rng.h"
#include "global/format.h"

#include <algorithm>
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor PolyZonotope::rand_point_impl(int64_t N, Rng &rng) const {
    const int64_t p = num_factors(), h = num_dependent(), q = GI_.shape()[1];
    // the factors of every point, and the monomial of every dependent generator
    std::vector<double> a(std::size_t(p * N)), b(std::size_t(q * N));
    rng.uniform(a.data(), a.size(), -1.0, 1.0);
    rng.uniform(b.data(), b.size(), -1.0, 1.0);
    std::vector<double> mono(std::size_t(h * N), 1.0);
    for (int64_t j = 0; j < h; ++j)
        for (int64_t i = 0; i < p; ++i)
            for (int64_t k = 0; k < N; ++k)
                mono[std::size_t(j * N + k)] *=
                    std::pow(a[std::size_t(i * N + k)], E_[std::size_t(j)][std::size_t(i)]);
    return c_ + G_.matmul(Tensor::like(c_, mono, {h, N})) + GI_.matmul(Tensor::like(c_, b, {q, N}));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
