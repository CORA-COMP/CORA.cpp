// enclose_points - a zonotope that contains a cloud of points
//
// "maiga" (Maiga et al. 2016) takes the best of 100 compressions of the cloud along its principal
// direction, by the trace of G'G; "stursberg" the box around the cloud in the frame of the
// principal axes of its covariance.
//
// Syntax:   Z = Zonotope::enclose_points(points);   Z = Zonotope::enclose_points(points,
// "stursberg");
//           Z = Zonotope::enclose_points(points, U);
// Inputs:   points - the points as columns (n, N)
//           method - "maiga" (default) or "stursberg"
//           U - an orthogonal frame (n, n), which gives the "stursberg" zonotope for any U
// Outputs:  Z - a zonotope with every point in it
// See also: enclose, interval
//
// CORA: contSet/@zonotope/enclosePoints.m
// CORA authors: Niklas Kochdumper
// Authors:       Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/zonotope/private/priv.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "global/format.h"

#include <algorithm>
#include <limits>
#include <optional>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

// Compression -------------------------------------------------------------------------------------

/// The box around the columns of X (n, N): its center and radius, as interval.enclosePoints.
Interval aux_box_of_points(const Tensor &X) { return {X.min(-1, true), X.max(-1, true)}; }

/// CORA's aux_compress: the columns of X moved towards the origin along g by at most |g|.
Tensor aux_compress(const Tensor &X, const Tensor &g) {
    const Tensor length = g.vector_norm(Tensor::Norm::two, -2, true);
    // g vanishes where the cloud is a single point or u is orthogonal to r; nothing to compress
    if (length.detach().item() == 0.0) return X;
    const Tensor u = g.divide(length);
    const Tensor d = u.matrix_transpose().matmul(X).minimum(length).maximum(length.negative());
    return X - u.matmul(d);
}

/// CORA's aux_cloud2zonotope: n compressions of the cloud at the ratio, then its box.
Zonotope aux_cloud_to_zonotope(Tensor X, double ratio) {
    const int64_t n = X.shape()[0];
    Tensor c = X.zeros_like({n, 1});
    std::vector<Tensor> generators;
    for (int64_t iter = 0; iter < n; ++iter) {
        const Interval box = aux_box_of_points(X);
        const Tensor mid = box.center(), r = box.rad();
        X = X - mid;
        const Tensor u = X.svd()[0].slice(-1, 0, 1);
        const Tensor g = u.multiply(u.matrix_transpose().matmul(r).abs() * ratio);
        X = aux_compress(X, g);
        c = c + mid;
        generators.push_back(g);
    }
    const Interval box = aux_box_of_points(X);
    generators.push_back(box.rad().diag());
    return {c + box.center(), Tensor::concat(generators, -1)};
}

// Checks ------------------------------------------------------------------------------------------

/// Why the points are refused, if they are: they must be a nonempty matrix of columns.
Refusal aux_refusal(const Tensor &points) {
    if (points.ndim() == 2 && points.shape()[0] >= 1 && points.shape()[1] >= 1) return std::nullopt;
    return std::pair{Error::wrong_value,
                     "the points must be a nonempty matrix (n, N) of columns, got the shape " +
                         shape_text(points.shape())};
}

} // namespace


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::enclose_points(const Tensor &points, const Tensor &U) {
    detail::Entry entry;
    const Refusal refusal = aux_refusal(points);
    CORA_CHECK(!refusal, refusal->first, [&] { return "enclose_points: " + refusal->second; });
    const int64_t n = points.shape()[0], N = points.shape()[1];
    // the frame must be square
    CORA_CHECK(U.ndim() == 2 && U.shape()[0] == n && U.shape()[1] == n, Error::dimension_mismatch,
               [&] {
                   return "enclose_points: the frame must be a matrix (" + std::to_string(n) +
                          ", " + std::to_string(n) + "), got the shape " + shape_text(U.shape());
               });
    // the box around the points in the frame U, as a zonotope
    const Tensor mean = points.sum(-1, true) / double(N);
    const Tensor sample = points - mean;
    const Tensor oriented = U.matrix_transpose().matmul(sample);
    const Tensor top = oriented.max(-1, true), bottom = oriented.min(-1, true);
    return {mean + U.matmul((top + bottom) / 2.0),
            U.multiply(((top - bottom) * 0.5).matrix_transpose())};
}

Zonotope Zonotope::enclose_points(const Tensor &points, const std::string &method) {
    detail::Entry entry;
    const Refusal refusal = aux_refusal(points);
    CORA_CHECK(!refusal, refusal->first, [&] { return "enclose_points: " + refusal->second; });
    // the two methods CORA has
    CORA_CHECK(method == "maiga" || method == "stursberg", Error::wrong_value, [&] {
        return "enclose_points: the method must be 'maiga' or 'stursberg', got '" + method + "'";
    });
    if (method == "stursberg") {
        // the principal axes of the covariance of the cloud
        const int64_t N = points.shape()[1];
        const Tensor sample = points - points.mean(-1, true);
        const Tensor covariance =
            sample.matmul(sample.matrix_transpose()) / double(std::max<int64_t>(N - 1, 1));
        return enclose_points(points, covariance.svd()[0]);
    }
    // "maiga": of the compressions at 100 ratios, the one with the least trace of G'G
    std::optional<Zonotope> best;
    double least = std::numeric_limits<double>::infinity();
    for (int i = 1; i <= 100; ++i) {
        const Zonotope Z = aux_cloud_to_zonotope(points, i / 100.0);
        const Tensor &G = Z.generators();
        const double trace = G.matrix_transpose().matmul(G).trace().detach().item();
        if (trace < least) {
            least = trace;
            best = Z;
        }
    }
    return *best;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
