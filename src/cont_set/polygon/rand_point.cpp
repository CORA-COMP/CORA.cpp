// rand_point - random points of a polygon, as CORA's randPoint_
//
// Uniform in the region: a triangle of the disjoint parts' fans drawn by area, then a point
// uniform in it (the reflected unit square). CORA samples the polyshape's triangulation alike.
//
// Syntax:   Tensor X = pgon.rand_point(N, rng);
// Inputs:   N - the number of points
//           rng - the generator
// Outputs:  X - the points as rows (N, 2)
// See also: contains, triangulation
//
// CORA: contSet/@polygon/randPoint_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polygon/private/priv.h"
#include "cora/cont_set/polygon/polygon.h"
#include "cora/global/rng.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor Polygon::rand_point_impl(int64_t N, Rng &rng) const {
    std::vector<std::array<Point, 3>> triangles;
    std::vector<double> cumulative;
    double total = 0;
    for (const Piece &part : priv_disjoint(priv_hosts(*this)))
        for (std::size_t i = 1; i + 1 < part.size(); ++i) {
            triangles.push_back({part[0], part[i], part[i + 1]});
            total += priv_area({part[0], part[i], part[i + 1]});
            cumulative.push_back(total);
        }
    if (triangles.empty())
        CORA_FAIL(Error::empty_set, "rand_point: the polygon has no area to sample");
    std::vector<double> u(3 * std::size_t(N)), out;
    rng.uniform(u.data(), u.size(), 0, 1);
    // a triangle by its area, then a point of the unit square folded into it
    for (int64_t k = 0; k < N; ++k) {
        const double pick = u[3 * k] * total;
        std::size_t t = 0;
        while (t + 1 < cumulative.size() && cumulative[t] < pick) ++t;
        double s = u[3 * k + 1], r = u[3 * k + 2];
        if (s + r > 1) s = 1 - s, r = 1 - r;
        const auto &[a, b, c] = triangles[t];
        out.insert(out.end(), {a[0] + s * (b[0] - a[0]) + r * (c[0] - a[0]),
                               a[1] + s * (b[1] - a[1]) + r * (c[1] - a[1])});
    }
    return Tensor::like(like_, out, {N, 2});
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
