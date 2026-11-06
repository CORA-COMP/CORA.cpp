// volume - the volume of a polytope, as CORA's polytope.volume_
//
// Inf if the set is unbounded, 0 if it is empty or flat. Else the pyramids over its facets from the
// mean of its vertices: volume = sum_i (distance_i / n) area_i, the area of a facet being the
// volume of its vertices in the hyperplane, by the same rule one dimension lower (CORA uses
// qhull's convhulln). Not a batch.
//
// Syntax:   double v = P.volume();
// Inputs:   -
// Outputs:  v - the n-dimensional volume
// See also: Polytope::is_bounded, Polytope::is_full_dim
//
// CORA: contSet/@polytope/volume_.m
// CORA authors: Viktor Kotsev, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"
#include "cora/global/within_tol.h"

#include <algorithm>
#include <cmath>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The k-dimensional volume of the hull of full-dimensional points (rows of k numbers).
double aux_hull_volume(const Tensor &like, const Rows &points, int64_t k) {
    if (k == 1) {
        double lo = points[0][0], hi = points[0][0];
        for (const Row &p : points) {
            lo = std::min(lo, p[0]);
            hi = std::max(hi, p[0]);
        }
        return hi - lo;
    }
    const HostH F = priv_constraints_of_points(like, points, k);
    if (!F.Ae.empty() || F.A.empty()) return 0.0;
    Row center(std::size_t(k), 0.0);
    for (const Row &p : points)
        for (std::size_t i = 0; i < center.size(); ++i) center[i] += p[i] / double(points.size());
    double volume = 0.0;
    for (std::size_t i = 0; i < F.A.size(); ++i) {
        // the vertices on the facet, in coordinates of its hyperplane
        const Rows frame = priv_gram_schmidt(F.A[i]);
        const Rows basis(frame.begin() + 1, frame.end());
        Rows facet;
        for (const Row &p : points) {
            if (!within_tol(priv_dot(F.A[i], p), F.b[i], numerics::kPolytopeFacet)) continue;
            Row q;
            for (const Row &u : basis) q.push_back(priv_dot(u, p));
            facet.push_back(q);
        }
        if (facet.size() < std::size_t(k)) continue;
        const double height = F.b[i] - priv_dot(F.A[i], center);
        volume += height * aux_hull_volume(like, facet, k - 1) / double(k);
    }
    return volume;
}

} // namespace


// ===========================================  MAIN  =========================================== //

double Polytope::volume() const {
    detail::Entry entry;
    const Tensor &like = is_h_rep() ? A_ : V_;
    CORA_CHECK(like.ndim() == 2, Error::not_supported,
               [] { return std::string("volume: it supports a single polytope, not a batch"); });
    if (priv_is_fullspace(*this) || !priv_polytope_is_bounded(*this))
        return std::numeric_limits<double>::infinity();
    if (priv_polytope_is_empty(*this) || !is_full_dim()) return 0.0;
    const Rows vertices = priv_polytope_vertices(*this);
    const int64_t n = dim();
    // fewer than n + 1 vertices span no volume
    if (int64_t(vertices.size()) <= n) return 0.0;
    return aux_hull_volume(like, vertices, n);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
