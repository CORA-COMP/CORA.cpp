// rand_point - random points of a capsule, as CORA's capsule.randPoint_
//
// "standard": a point of the axis plus a point of the ball (not uniform in volume). "extreme": the
// boundary point in a random direction d, c + g sign(d'g) + r d / |d|.
//
// Syntax:   p = C.rand_point(N, rng);   p = C.rand_point(N, rng, type);
// Inputs:   N - the number of points
//           rng - the generator of the random numbers
//           type - "standard" (default) or "extreme"
// Outputs:  p - the points as columns (..., n, N)
// See also: Zonotope::rand_point
//
// CORA: contSet/@capsule/randPoint_.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/capsule/private/priv.h"
#include "cora/cont_set/capsule/capsule.h"
#include "cora/global/rng.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Capsule::rand_point(int64_t N, Rng &rng, const std::string &type) const {
    detail::Entry entry;
    // the count and the type are checked here, the sampling is rand_point_impl
    CORA_CHECK(N >= 0, Error::wrong_value, [&] {
        return "rand_point: the number of points must be nonnegative, got " + std::to_string(N);
    });
    CORA_CHECK(type == "standard" || type == "extreme", Error::no_specific_alg, [&] {
        return "rand_point: unknown type '" + type + "'; use \"standard\" or \"extreme\"";
    });
    return rand_point_impl(N, rng, type);
}

Tensor Capsule::rand_point_impl(int64_t N, Rng &rng) const {
    return rand_point_impl(N, rng, "standard");
}

Tensor Capsule::rand_point_impl(int64_t N, Rng &rng, const std::string &type) const {
    if (is_empty()) CORA_FAIL(Error::empty_set, "rand_point: the capsule is empty");
    // random numbers in the shape (..., k, N), drawn on the host and moved to the tensor's backend
    const auto draw = [&](int64_t rows, double low) {
        Shape shape = c_.shape();
        shape[shape.size() - 2] = rows;
        shape.back() = N;
        std::vector<double> values(static_cast<std::size_t>(shape.numel()));
        rng.uniform(values.data(), values.size(), low, 1.0);
        return Tensor::like(c_, values, shape);
    };
    // a random direction in the unit cube, normalized
    const Tensor d = draw(dim(), -1.0);
    const Tensor direction = d.divide(priv_norm(d));
    if (type == "standard") {
        const Tensor axis = g_.multiply(draw(1, -1.0)), ball = r_.multiply(draw(1, 0.0));
        return c_.add(axis).add(direction.multiply(ball));
    }
    if (type == "extreme") {
        const Tensor along = g_.matrix_transpose().matmul(direction).sign();
        return c_.add(g_.multiply(along)).add(direction.multiply(r_));
    }
    CORA_FAIL(Error::no_specific_alg, "rand_point: unknown type '" + type + "'");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
