// randPoint - random points of a zonotope, as CORA's zonotope.randPoint
//
// Syntax:   P = Z.randPoint(N, rng);   P = Z.randPoint(N, rng, "extreme");
// Inputs:   N - number of points; rng - random numbers (a seed makes them repeatable)
//           type - "standard": c + G b, b ~ U[-1, 1]^m;  "extreme": b in {-1, 1}^m
// Outputs:  P - points as columns (..., n, N)
// See also: Interval::randPoint, LinearSys::simulateRandom

#include "contSet/zonotope/zonotope.h"

#include "global/rng.h"

#include <stdexcept>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::ct {

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
        for (double &v : b) v = v < 0.0 ? -1.0 : 1.0;  // snap to the corners
    return Tensor::like(like, b, shape);
}

} // namespace

// ===========================================  MAIN  =========================================== //

Tensor Zonotope::randPoint(int64_t N, Rng &rng, const std::string &type) const {
    if (type == "standard") return c + G.matmul(aux_factors(c, G, N, rng, false));
    if (type == "extreme") return c + G.matmul(aux_factors(c, G, N, rng, true));
    throw std::invalid_argument("Zonotope::randPoint: unknown type '" + type +
                                "'; use \"standard\" or \"extreme\"");
}

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
