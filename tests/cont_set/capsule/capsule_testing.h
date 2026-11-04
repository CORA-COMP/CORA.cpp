// capsule_testing - what the tests of the capsule share: random capsules and brute-force references
//
// Syntax:   Capsule C = test::random_capsule(rng, n);   Tensor P = test::sample_points(C, rng, N);
//           double d = test::axis_distance(C, p);
// Sets:     the center and the generator are standard normal, the radius uniform in [0.2, 1]
// See also: testing.h

#pragma once

#include "cora/cont_set/capsule/capsule.h"
#include "cora/cont_set/empty_set/empty_set.h"
#include "cora/cont_set/fullspace/fullspace.h"
#include "cora/cont_set/registry.h"
#include "testing.h"

#include <cmath>

namespace test {

/// A random capsule of n dimensions.
inline cora::Capsule random_capsule(cora::Rng &rng, int n) {
    std::vector<double> c(n), g(n), r(1);
    rng.normal(c.data(), c.size(), 1.0);
    rng.normal(g.data(), g.size(), 1.0);
    rng.uniform(r.data(), 1, 0.2, 1.0);
    return {column(c), column(g), r[0]};
}

/// The distance of the point p (host values) from the segment c - g .. c + g, found by a fine grid
/// over the parameter plus one refinement (no formula of the library).
inline double axis_distance(const cora::Capsule &C, const std::vector<double> &p) {
    const std::vector<double> c = C.c().data(), g = C.generator().data();
    const auto distance = [&](double t) {
        double sum = 0;
        for (std::size_t i = 0; i < c.size(); ++i) {
            const double d = c[i] + t * g[i] - p[i];
            sum += d * d;
        }
        return std::sqrt(sum);
    };
    double best = 1e300, at = 0;
    for (int k = 0; k <= 2000; ++k) {
        const double t = -1.0 + 2.0 * k / 2000.0, d = distance(t);
        if (d < best) best = d, at = t;
    }
    for (int k = -100; k <= 100; ++k)
        best = std::min(best, distance(std::max(-1.0, std::min(1.0, at + k * 1e-5))));
    return best;
}

/// N points of the capsule as columns (n, N): a point of the axis plus a point of the ball.
inline cora::Tensor sample_points(const cora::Capsule &C, cora::Rng &rng, int N) {
    const int64_t n = C.dim();
    const std::vector<double> c = C.c().data(), g = C.generator().data();
    const double r = C.r().data()[0];
    std::vector<double> points(std::size_t(n) * N), direction(n), t(1), radius(1);
    for (int k = 0; k < N; ++k) {
        rng.normal(direction.data(), n, 1.0);
        rng.uniform(t.data(), 1, -1.0, 1.0);
        rng.uniform(radius.data(), 1, 0.0, 1.0);
        double length = 0;
        for (const double d : direction) length += d * d;
        // a point of the ball of radius r (not uniform in volume)
        const double scale = r * radius[0] / std::sqrt(length);
        for (int64_t i = 0; i < n; ++i)
            points[i * N + k] = c[i] + t[0] * g[i] + scale * direction[i];
    }
    return cora::Tensor::from_data(points, {n, int64_t(N)});
}

} // namespace test
