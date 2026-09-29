// randPoint - random points of a box, as CORA's interval.randPoint
//
// Syntax:   P = I.randPoint(N, rng);
// Inputs:   N - number of points; rng - random numbers (a seed makes them repeatable)
// Outputs:  P - points uniform in the box, columns (..., n, N)
// See also: Zonotope::randPoint

#include "contSet/interval/interval.h"

#include "global/rng.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora::ct {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Fractions U[0, 1] of shape (..., n, N), on the backend and device of `like`.
Tensor aux_fractions(const Tensor &like, int64_t N, Rng &rng) {
    std::vector<int64_t> shape = like.shape();
    shape.back() = N;
    int64_t count = 1;
    for (const int64_t d : shape) count *= d;
    std::vector<double> u(count);
    rng.uniform(u.data(), u.size(), 0.0, 1.0);
    return Tensor::like(like, u, shape);
}

} // namespace

// ===========================================  MAIN  =========================================== //

Tensor Interval::randPoint(int64_t N, Rng &rng) const {
    // inf + diag(sup - inf) * u scales row i of the fractions u by the width of side i.
    return inf + (sup - inf).diag().matmul(aux_fractions(inf, N, rng));
}

} // namespace cora::ct

// ---------------------------------------  END OF CODE  ---------------------------------------- //
