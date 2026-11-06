// hausdorff_distance - the Hausdorff distance of a polytope to a polytope, a box or points
//
// Syntax:   double d = P.hausdorff_distance(P2);   d = P.hausdorff_distance(I);
//           d = P.hausdorff_distance(points);
// Inputs:   P2, I - a polytope or a box of the dimension of P;  points - the columns (n, N)
// Outputs:  d - the Euclidean Hausdorff distance: the largest distance of a vertex of one set to
//           the other set, each a quadratic program (a least-distance problem). Inf if a set is
//           unbounded and the other is not or they differ, or if exactly one is empty (the empty
//           set to the empty set, and an unbounded P to no points, are 0 and Inf as in CORA).
//           Two unbounded sets are at distance 0 if each holds the other (up to 1e-10). Not a
//           batch
// See also: Polytope::contains, Zonotope::hausdorff_distance_to_box
//
// CORA: contSet/@polytope/hausdorffDist.m
//
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"
#include "cora/global/tensor/tolerance.h"
#include "global/format.h"

#include <algorithm>
#include <limits>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

constexpr double kInf = std::numeric_limits<double>::infinity();

/// The largest distance of the points to the polytope given by H.
double aux_farthest(const HostH &H, const Rows &points) {
    double value = 0.0;
    for (const Row &x : points) value = std::max(value, priv_distance_to_polytope(H, x));
    return value;
}

} // namespace


// ===========================================  MAIN  =========================================== //

double Polytope::hausdorff_distance(const Tensor &points) const {
    detail::Entry entry;
    CORA_CHECK(points.defined() && points.ndim() == 2 && points.shape()[0] == dim(),
               Error::dimension_mismatch, [&] {
                   return "hausdorff_distance: the points must be a matrix (" +
                          std::to_string(dim()) + ", N) of columns, got the shape " +
                          shape_text(points.shape());
               });
    if (!priv_polytope_is_bounded(*this)) return kInf;
    if (priv_polytope_is_empty(*this)) return points.shape()[1] == 0 ? 0.0 : kInf;
    return aux_farthest(priv_host_h(constraints()), priv_host_points(points));
}

double Polytope::hausdorff_distance(const Polytope &P2) const {
    detail::Entry entry;
    CORA_CHECK(P2.dim() == dim(), Error::dimension_mismatch, [&] {
        return "hausdorff_distance: the sets differ in dimension, " + std::to_string(dim()) +
               " and " + std::to_string(P2.dim());
    });
    const bool empty = priv_polytope_is_empty(*this), other_empty = priv_polytope_is_empty(P2);
    if (empty) return other_empty ? 0.0 : kInf;
    if (other_empty) return kInf;
    if (!priv_polytope_is_bounded(*this)) {
        if (priv_polytope_is_bounded(P2)) return kInf;
        // two unbounded sets: 0 if they are the same
        return contains(P2, "exact", numerics::kPolytopeHausdorff) &&
                       P2.contains(*this, "exact", numerics::kPolytopeHausdorff)
                   ? 0.0
                   : kInf;
    }
    if (!priv_polytope_is_bounded(P2)) return kInf;
    // the farthest vertex of each set from the other
    const HostH H1 = priv_host_h(constraints()), H2 = priv_host_h(P2.constraints());
    return std::max(aux_farthest(H2, priv_polytope_vertices(*this)),
                    aux_farthest(H1, priv_polytope_vertices(P2)));
}

double Polytope::hausdorff_distance(const Interval &I) const {
    detail::Entry entry;
    CORA_CHECK(I.dim() == dim(), Error::dimension_mismatch, [&] {
        return "hausdorff_distance: the sets differ in dimension, " + std::to_string(dim()) +
               " and " + std::to_string(I.dim());
    });
    return hausdorff_distance(priv_polytope_of_interval(I));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
