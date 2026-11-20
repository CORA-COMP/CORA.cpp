// rand_point - random points of a zonotope, as CORA's zonotope.randPoint
//
// Syntax:   P = Z.rand_point(N, rng);   P = Z.rand_point(N, rng, "extreme");
// Inputs:   N - number of points
//           rng - random numbers (a seed makes them repeatable)
//           type - "standard": c + G b, b ~ U[-1, 1]^m;  "extreme": b in {-1, 1}^m
// Outputs:  P - points as rows (..., N, n)
// See also: Interval::rand_point, LinearSys::simulate_random
//
// CORA: contSet/@zonotope/randPoint_.m
// CORA authors: Matthias Althoff, Mark Wetzlinger, Adrian Kulmburg, Severin Prenitzer
// Authors:       Tobias Ladner, Claude Opus 5.5
// Written:       02-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/cont_set/private/priv_rand_point.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/error.h"

#include "cora/global/rng.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The factors b, one column per point, (..., m, N); the batch dimensions are those of G.
Tensor aux_factors(const Tensor &like, const Tensor &G, int64_t N, Rng &rng, bool extreme) {
    std::vector<int64_t> shape = G.shape();
    shape[shape.size() - 2] = shape.back();
    shape.back() = N;
    int64_t count = 1;
    for (const int64_t d : shape) count *= d;

    std::vector<double> b(count);
    rng.uniform(b.data(), b.size(), -1.0, 1.0);
    if (extreme)
        for (double &v : b) v = v < 0.0 ? -1.0 : 1.0; // snap to the corners
    return Tensor::like(like, b, shape);
}

} // namespace


// ===========================================  MAIN  =========================================== //

Tensor Zonotope::rand_point(int64_t N, Rng &rng, const std::string &type) const {
    detail::Entry entry;
    const detail::Refusal refusal = priv_rand_point_refusal(N, type);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return rand_point_impl(N, rng, type);
}

Tensor Zonotope::rand_point_impl(int64_t N, Rng &rng) const {
    return rand_point_impl(N, rng, "standard");
}

Tensor Zonotope::rand_point_impl(int64_t N, Rng &rng, const std::string &type) const {
    // CORA's columns c + G b (..., n, N), drawn in CORA's order, as rows (..., N, n)
    const Tensor c = c_.expand_dims(-1), G = G_.matrix_transpose();
    if (type == "standard")
        return (c + G.matmul(aux_factors(c, G, N, rng, false))).matrix_transpose();
    if (type == "extreme")
        return (c + G.matmul(aux_factors(c, G, N, rng, true))).matrix_transpose();


    CORA_FAIL(Error::wrong_value,
              "rand_point: unknown type '" + type + "'; use \"standard\" or \"extreme\"");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
