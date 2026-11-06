// rand_point - random points of a polytope, as CORA's polytope.randPoint_
//
// "standard" (default) takes a random convex combination of n + 1 extreme points, "extreme" an
// extreme point in a random direction (for an unbounded set along a random row),
// "uniform:hitAndRun" the walk of hit and run from a "standard" point, for full-dimensional sets.
// R^n gets Gaussian points, the empty set none (n, 0). The other uniform walks of CORA (billiard,
// ball) are Error::not_supported. The points are drawn from rng, so they differ from MATLAB's. Not
// a batch.
//
// Syntax:   p = P.rand_point(N, rng);   p = P.rand_point(N, rng, "extreme");
// Inputs:   N - the number of points, a positive integer
//           rng - the random numbers
//           type - "standard" (default), "extreme", "uniform:hitAndRun"
// Outputs:  p - N points, the columns of (n, N)
// See also: Polytope::support_function, Polytope::is_full_dim
//
// CORA: contSet/@polytope/randPoint_.m
// CORA authors: Mark Wetzlinger, Adrian Kulmburg
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"

#include <algorithm>
#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

// The host form -----------------------------------------------------------------------------------

/// The set on the host: its halfspaces (a single set), and its vertices if it has the vertex form.
struct Host {
    HostH H;
    Rows vertices;
    bool vertex_form = false;
};

/// The point of the set that maximizes d'x (the first one of the vertices, a linear program else).
Row aux_support_point(const Host &host, const Row &d) {
    if (host.vertex_form) {
        std::size_t best = 0;
        for (std::size_t k = 1; k < host.vertices.size(); ++k)
            if (priv_dot(d, host.vertices[k]) > priv_dot(d, host.vertices[best])) best = k;
        return host.vertices[best];
    }
    return priv_support(host.H, d, true).x;
}

// The samplers ------------------------------------------------------------------------------------

/// An extreme point in a random direction; for an unbounded set along a random row of [A; Ae; -Ae].
Row aux_extreme(const Host &host, bool bounded, Rng &rng) {
    const std::size_t n = std::size_t(host.H.n);
    Row d(n);
    if (!bounded) {
        const Rows rows = priv_equality_to_inequality(host.H).first;
        double pick = 0.0;
        rng.uniform(&pick, 1, 0.0, double(rows.size()));
        return aux_support_point(host, rows[std::min(rows.size() - 1, std::size_t(pick))]);
    }
    rng.uniform(d.data(), n, -0.5, 0.5);
    const double norm = priv_norm(d);
    for (double &x : d) x /= norm;
    return aux_support_point(host, d);
}

/// A random convex combination of n + 1 extreme points.
Row aux_standard(const Host &host, bool bounded, Rng &rng) {
    const std::size_t n = std::size_t(host.H.n);
    Row weights(n + 1), point(n, 0.0);
    rng.uniform(weights.data(), weights.size(), 0.0, 1.0);
    double sum = 0.0;
    for (const double w : weights) sum += w;
    for (const double w : weights) {
        const Row extreme = aux_extreme(host, bounded, rng);
        for (std::size_t i = 0; i < n; ++i) point[i] += extreme[i] * w / sum;
    }
    return point;
}

/// N points of hit and run: a random direction, a uniform point on the chord through the current
/// point (CORA's aux_randPointHitAndRun for a full-dimensional set).
Rows aux_hit_and_run(const Host &host, int64_t N, Rng &rng) {
    const HostH &H = host.H;
    const std::size_t n = std::size_t(H.n);
    Row current = aux_standard(host, true, rng);
    Rows points;
    for (int64_t k = 0; k < N; ++k) {
        Row direction(n);
        rng.normal(direction.data(), n, 1.0);
        const double norm = priv_norm(direction);
        for (double &x : direction) x /= norm;
        // the chord: the rows bound the step from above (A d > 0) and from below (A d < 0)
        double upper = kInf, lower = -kInf;
        for (std::size_t i = 0; i < H.A.size(); ++i) {
            const double slope = priv_dot(H.A[i], direction);
            const double step = (H.b[i] - priv_dot(H.A[i], current)) / slope;
            if (slope > 0) upper = std::min(upper, step);
            if (slope < 0) lower = std::max(lower, step);
        }
        if (!std::isfinite(upper) || !std::isfinite(lower))
            CORA_FAIL(Error::special_error,
                      "The polytope is either empty or unbounded, therefore one cannot sample it "
                      "uniformly using the hit-and-run algorithm.");
        if (upper < lower) upper = lower = 0.0;
        double fraction = 0.0;
        rng.uniform(&fraction, 1, 0.0, 1.0);
        const double step = lower + fraction * (upper - lower);
        for (std::size_t i = 0; i < n; ++i) current[i] += step * direction[i];
        points.push_back(current);
    }
    return points;
}

/// The (n, N) tensor of the points, on the backend of `like`.
Tensor aux_matrix(const Tensor &like, int64_t n, const Rows &points) {
    return priv_points_tensor(like, n, points);
}

} // namespace


// ===========================================  MAIN  =========================================== //

Tensor Polytope::rand_point_impl(int64_t N, Rng &rng) const {
    return rand_point(N, rng, "standard");
}

Tensor Polytope::rand_point(int64_t N, Rng &rng, const std::string &type) const {
    detail::Entry entry;
    // the count and the type are checked before the set is read
    CORA_CHECK(N >= 1, Error::wrong_value, [&] {
        return "rand_point: the number of points must be a positive integer, got " +
               std::to_string(N);
    });
    CORA_CHECK(type == "standard" || type == "extreme" || type == "uniform:hitAndRun" ||
                   type == "uniform" || type == "uniform:billiardWalk" ||
                   type == "uniform:ballWalk",
               Error::no_specific_alg,
               [&] { return "rand_point: no algorithm '" + type + "' for a polytope"; });
    CORA_CHECK(
        type == "standard" || type == "extreme" || type == "uniform:hitAndRun",
        Error::not_supported,
        [&] { return "rand_point: the walk '" + type + "' is not ported; use uniform:hitAndRun"; });
    const Tensor &like = is_h_rep() ? A_ : V_;
    CORA_CHECK(like.ndim() == 2, Error::not_supported, [] {
        return std::string("rand_point: it supports a single polytope, not a batch");
    });
    const int64_t n = dim();
    // R^n: Gaussian points; the empty set: none
    if (priv_is_fullspace(*this)) {
        std::vector<double> flat(std::size_t(n * N));
        rng.normal(flat.data(), flat.size(), 1.0);
        return Tensor::like(like, flat, {n, N});
    }
    if (priv_polytope_is_empty(*this)) return priv_zeros(like, n, 0);
    // the host form of the set, then the sampler of the type
    Host host;
    host.H = priv_host_h(constraints());
    host.vertex_form = is_v_rep();
    if (host.vertex_form) host.vertices = priv_polytope_vertices(*this);
    const bool bounded = priv_polytope_is_bounded(*this);
    if (type == "uniform:hitAndRun") {
        CORA_CHECK(is_full_dim() && bounded, Error::special_error, [] {
            return std::string("The hit-and-run walk needs a bounded polytope with an interior.");
        });
        return aux_matrix(like, n, aux_hit_and_run(host, N, rng));
    }
    Rows points;
    for (int64_t k = 0; k < N; ++k)
        points.push_back(type == "standard" ? aux_standard(host, bounded, rng)
                                            : aux_extreme(host, bounded, rng));
    return aux_matrix(like, n, points);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
