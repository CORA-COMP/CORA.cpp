// contains - whether a capsule contains points, a capsule, a zonotope or a box
//
// Syntax:   res = C.contains(p);   res = C.contains(S, tol);
// Inputs:   p - points (..., n, N)
//           S, Z, I - a capsule, a zonotope or a box of the dimension of C
//           tol - how far a point may lie outside, in the distance to the axis (absolutely or
//                 relative to r, as within_tol); default: the repository tolerance
//                 (docs/architecture.md)
// Outputs:  res - true if all points are within r of the axis. A capsule is inside if the two balls
//           at the ends of its axis are: their centers are within r - S.r of the axis. A zonotope
//           or a box is inside exactly if its corners are (at most 12 generators). The empty
//           capsule contains only the empty set, and every capsule the empty one. Of a batch:
//           every member. CORA's second and third outputs (the certificate and the scaling) are
//           not returned
// See also: is_intersecting, Zonotope::contains
//
// CORA: contSet/@capsule/contains_.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/capsule/private/priv.h"
#include "cont_set/zonotope/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Why the set is refused, if it is: another dimension, or a zonotope with too many corners.
detail::Refusal aux_zonotope_refusal(const Capsule &C, const Zonotope &Z) {
    if (Z.dim() != C.dim())
        return std::pair{Error::dimension_mismatch, "the sets differ in dimension, " +
                                                        std::to_string(C.dim()) + " and " +
                                                        std::to_string(Z.dim())};
    if (Z.compact().generators().shape().back() > kMaxCornerGenerators)
        return std::pair{Error::not_supported,
                         std::string("a set of more than 12 generators has too many corners to "
                                     "test one by one")};
    return std::nullopt;
}

/// Whether the points p (..., n, N) are all within r of the axis of (c, g), the capsule being
/// nonempty.
bool aux_points_inside(const Capsule &C, const Tensor &p, double tol) {
    if (p.size(-1) == 0) return true;
    const Tensor distance = priv_axis_distance(C.c(), C.generator(), p);
    const Tensor inside = priv_at_most(distance, C.r(), tol);
    return priv_all((1.0 - priv_empty_mask(C.r())).multiply(inside).min(-1, true));
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool Capsule::contains(const Tensor &p, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, c_.dtype(), "contains");
    const detail::Refusal refusal = priv_points_refusal(*this, p);
    CORA_CHECK(!refusal, refusal->first, [&] { return "contains: " + refusal->second; });
    return aux_points_inside(*this, p, tol);
}

bool Capsule::contains(const Capsule &S, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, c_.dtype(), "contains");
    CORA_CHECK(S.dim() == dim(), Error::dimension_mismatch, [&] {
        return "contains: the capsules differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(S.dim());
    });
    // the ends of the axis of S, each with the radius of S
    const Tensor ends = Tensor::concat({S.c_.add(S.g_), S.c_.subtract(S.g_)}, -1);
    const Tensor reach = priv_axis_distance(c_, g_, ends).add(S.r_);
    const Tensor inside = priv_at_most(reach, r_, tol).min(-1, true);
    // the empty capsule is inside every capsule, and holds only the empty one
    const Tensor mine = priv_empty_mask(r_), theirs = priv_empty_mask(S.r_);
    return priv_all(theirs.maximum((1.0 - mine).multiply(inside)));
}

bool Capsule::contains(const Zonotope &Z, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, c_.dtype(), "contains");
    const detail::Refusal refusal = aux_zonotope_refusal(*this, Z);
    CORA_CHECK(!refusal, refusal->first, [&] { return "contains: " + refusal->second; });
    // a convex set holds a zonotope exactly if it holds its corners
    return aux_points_inside(*this, priv_corners(Z.compact()), tol);
}

bool Capsule::contains(const Interval &I, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, c_.dtype(), "contains");
    CORA_CHECK(I.dim() == dim() && (I.is_empty() || I.inf().shape().back() == 1),
               Error::dimension_mismatch, [&] {
                   return "contains: the interval must be a box of dimension " +
                          std::to_string(dim()) + ", got the shape " + shape_text(I.inf().shape());
               });
    // the empty box is contained in every set
    if (I.is_empty()) return true;
    // a box is the zonotope of its axes
    const Zonotope Z = I.zonotope();
    const detail::Refusal refusal = aux_zonotope_refusal(*this, Z);
    CORA_CHECK(!refusal, refusal->first, [&] { return "contains: " + refusal->second; });
    return aux_points_inside(*this, priv_corners(Z.compact()), tol);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
