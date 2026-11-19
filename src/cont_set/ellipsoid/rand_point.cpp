// rand_point - random points of an ellipsoid, as CORA's ellipsoid.randPoint_
//
// On the range of Q: q + G u for the factor G (n, r) and u uniform in the r-ball ("standard",
// "uniform") or on its sphere ("extreme"). A point repeats its center. Not a batch.
//
// Syntax:   p = E.rand_point(N, rng);   p = E.rand_point(N, rng, type);
// Inputs:   N - the number of points
//           rng - the generator of the random numbers
//           type - "standard" (default), "uniform" or "extreme"
// Outputs:  p - the points as rows (N, n)
// See also: Zonotope::rand_point
//
// CORA: contSet/@ellipsoid/randPoint_.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       06-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "cora/global/rng.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Ellipsoid::rand_point(int64_t N, Rng &rng, const std::string &type) const {
    detail::Entry entry;
    // the count and the type are checked here, the sampling is rand_point_impl
    CORA_CHECK(N >= 0, Error::wrong_value, [&] {
        return "rand_point: the number of points must be nonnegative, got " + std::to_string(N);
    });
    CORA_CHECK(type == "standard" || type == "uniform" || type == "extreme", Error::no_specific_alg,
               [&] {
                   return "rand_point: unknown type '" + type +
                          "'; use \"standard\", \"uniform\" or \"extreme\"";
               });
    return rand_point_impl(N, rng, type);
}

Tensor Ellipsoid::rand_point_impl(int64_t N, Rng &rng) const {
    return rand_point_impl(N, rng, "standard");
}

Tensor Ellipsoid::rand_point_impl(int64_t N, Rng &rng, const std::string &type) const {
    // the empty set has no points, and the rank of one ellipsoid decides the dimension of the ball
    if (priv_is_empty(*this)) CORA_FAIL(Error::empty_set, "rand_point: the ellipsoid is empty");
    if (Q_.ndim() != 2)
        CORA_FAIL(Error::not_supported, "rand_point: it supports a single ellipsoid, not a batch");
    // CORA's columns (n, N), as rows (N, n)
    const Tensor q = q_.expand_dims(-1);
    if (priv_is_point(*this)) return q.tile({1, N}).matrix_transpose();
    const Tensor G = priv_range_factor(*this);
    const int64_t rank = G.shape().back();
    if (rank == 0) return q.tile({1, N}).matrix_transpose();
    // directions: Gaussian columns of unit length
    std::vector<double> x(static_cast<std::size_t>(rank * N));
    rng.normal(x.data(), x.size(), 1.0);
    Tensor u = Tensor::like(q_, x, {rank, N});
    u = u.divide(u.square().sum(0, true).sqrt());
    if (type != "extreme") {
        // radii r^(1/rank) for r uniform: uniform in the volume of the ball
        std::vector<double> radius(static_cast<std::size_t>(N));
        rng.uniform(radius.data(), radius.size(), 0.0, 1.0);
        u = u.multiply(Tensor::like(q_, radius, {1, N}).pow(1.0 / static_cast<double>(rank)));
    }
    return q.add(G.matmul(u)).matrix_transpose();
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
