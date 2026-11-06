// distance - the distance of a polytope to a polytope or points, as CORA's polytope.distance
//
// Inf if either set is empty, 0 if they meet (up to 1e-12). The distance of two polytopes is the
// distance of the origin to P + (-S), a polytope, by the least-distance program of
// hausdorff_distance (CORA solves a quadratic program in both sets' variables); of a point to the
// set the same program. A hyperplane is no special case (CORA divides by the squared norm of its
// normal, which is the distance only for a unit normal). Not a batch.
//
// Syntax:   d = P.distance(S);   Tensor d = P.distance(points);
// Inputs:   S - a polytope of the dimension of P
//           points - a matrix (n, N)
// Outputs:  d - the distance, a number or a row (1, N)
// See also: Polytope::hausdorff_distance, Polytope::is_intersecting
//
// CORA: contSet/@polytope/distance.m
// CORA authors: Victor Gassmann, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"
#include "global/format.h"

#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

/// Why the call is refused: a batch.
detail::Refusal aux_refusal(const Polytope &P) {
    if ((P.is_h_rep() ? P.A() : P.V()).ndim() != 2)
        return std::pair{Error::not_supported,
                         std::string("distance: it supports a single polytope, not a batch")};
    return std::nullopt;
}

} // namespace


// ===========================================  MAIN  =========================================== //

double Polytope::distance(const Polytope &S) const {
    detail::Entry entry;
    const detail::Refusal refusal = aux_refusal(*this);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    CORA_CHECK(S.dim() == dim(), Error::dimension_mismatch, [&] {
        return "distance: the sets differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(S.dim());
    });
    if (priv_polytope_is_empty(*this) || priv_polytope_is_empty(S)) return kInf;
    if (is_intersecting(S, "exact", numerics::kPolytopeDistance)) return 0.0;
    // the origin against {x - y | x in P, y in S}
    const Tensor &like = is_h_rep() ? A_ : V_;
    const Polytope difference = plus(S.linear_map(priv_identity(like, dim()) * -1.0));
    return priv_distance_to_polytope(priv_host_h(difference.constraints()),
                                     Row(std::size_t(dim()), 0.0));
}

Tensor Polytope::distance(const Tensor &points) const {
    detail::Entry entry;
    const detail::Refusal refusal = aux_refusal(*this);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    CORA_CHECK(points.defined() && points.ndim() == 2 && points.shape()[0] == dim(),
               Error::dimension_mismatch, [&] {
                   return "distance: the points must be a matrix (" + std::to_string(dim()) +
                          ", N) of columns, got the shape " +
                          shape_text(points.defined() ? points.shape() : Shape{});
               });
    // zero for a point in the set, else the least-distance program; none for the empty set
    const Tensor &like = is_h_rep() ? A_ : V_;
    const int64_t count = points.shape()[1];
    Row distances;
    const bool empty = priv_polytope_is_empty(*this);
    const HostH H = empty ? HostH{} : priv_host_h(constraints());
    for (const Row &y : priv_host_points(points)) {
        // a point in the set has distance 0
        const Tensor column = priv_column_tensor(like, y);
        distances.push_back(empty ? kInf
                            : contains(column, "exact", numerics::kPolytopeDistance)
                                ? 0.0
                                : priv_distance_to_polytope(H, y));
    }
    return Tensor::like(like, distances, {1, count});
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
