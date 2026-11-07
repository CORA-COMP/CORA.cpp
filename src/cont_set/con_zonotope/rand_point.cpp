// rand_point - random points of a constrained zonotope, as CORA's conZonotope.randPoint
//
// Without constraints the zonotope's. "standard" mixes the candidate vertices with random convex
// weights; "extreme" takes the support point of random directions. CORA's "uniform" walks are not
// ported. A set with constraints is not a batch.
//
// Syntax:   P = cz.rand_point(N, rng);   P = cz.rand_point(N, rng, "extreme");
// Inputs:   N - number of points
//           rng - random numbers
//           type - "standard" or "extreme"
// Outputs:  P - points as columns (n, N); none for an empty set
// See also: support_function, vertices
//
// CORA: contSet/@conZonotope/randPoint_.m
// CORA authors: Niklas Kochdumper, Severin Prenitzer, Adrian Kulmburg
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "cont_set/cont_set/private/priv_rand_point.h"
#include "cora/global/rng.h"
#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// A random convex combination of the columns of V (n, K) for each of N points.
Tensor aux_mixed(const Tensor &V, int64_t N, Rng &rng) {
    const int64_t K = V.shape().back();
    std::vector<double> weights(static_cast<std::size_t>(K * N));
    rng.uniform(weights.data(), weights.size(), 0.0, 1.0);
    for (int64_t j = 0; j < N; ++j) {
        double total = 0.0;
        for (int64_t k = 0; k < K; ++k) total += weights[std::size_t(k * N + j)];
        for (int64_t k = 0; k < K; ++k) weights[std::size_t(k * N + j)] /= total;
    }
    return V.matmul(Tensor::like(V, weights, {K, N}));
}

/// The support points of N random directions, as columns.
Tensor aux_extreme(const ConZonotope &cz, int64_t N, Rng &rng) {
    const HostConZonotope H = priv_host_con_zonotope(cz);
    std::vector<double> points;
    points.reserve(std::size_t(H.n * N));
    for (int64_t j = 0; j < N; ++j) {
        std::vector<double> d(static_cast<std::size_t>(H.n));
        rng.uniform(d.data(), d.size(), -0.5, 0.5);
        double norm = 0.0;
        for (const double v : d) norm += v * v;
        for (double &v : d) v /= std::sqrt(norm);
        const std::optional<PrivSupport> best = priv_support(H, d);
        if (!best) return cz.c().zeros_like({H.n, 0});
        points.insert(points.end(), best->point.begin(), best->point.end());
    }
    // the points were listed one after the other: transpose to columns
    std::vector<double> columns(points.size());
    for (int64_t j = 0; j < N; ++j)
        for (int64_t i = 0; i < H.n; ++i)
            columns[std::size_t(i * N + j)] = points[std::size_t(j * H.n + i)];
    return Tensor::like(cz.c(), columns, {H.n, N});
}

} // namespace


// ===========================================  MAIN  =========================================== //

Tensor ConZonotope::rand_point(int64_t N, Rng &rng, const std::string &type) const {
    detail::Entry entry;
    const detail::Refusal refusal = priv_rand_point_refusal(N, type);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return rand_point_impl(N, rng, type);
}

Tensor ConZonotope::rand_point_impl(int64_t N, Rng &rng) const {
    return rand_point_impl(N, rng, "standard");
}

Tensor ConZonotope::rand_point_impl(int64_t N, Rng &rng, const std::string &type) const {
    if (A_.shape()[A_.ndim() - 2] == 0) return Zonotope(c_, G_).rand_point(N, rng, type);
    if (type == "extreme") return aux_extreme(*this, N, rng);
    // the candidates are points of the set, and so are their convex combinations
    const Tensor V = priv_vertex_candidates(*this);
    if (V.shape().back() == 0) return V;
    return aux_mixed(V, N, rng);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
