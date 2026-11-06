// is_intersecting - whether a zonotope and a zonotope, a box or points intersect
//
// Syntax:   res = Z.is_intersecting(S);   res = Z.is_intersecting(S, "approx", tol);
// Inputs:   S - a zonotope, a box, or points (n, N), of the dimension of Z
//           type - "exact" (default) decides, by a linear program; "approx" tests only the boxes
//                  around the two sets, so it can say true for disjoint sets
//           tol - S enlarged by tol in every coordinate (absolute) is intersected; for points,
//                 the tolerance of contains; default: the repository tolerance
//                 (docs/architecture.md). Not a batch
// Outputs:  res - true if the sets intersect. For points all of them must lie in Z
//           (CORA's call to contains_ has a missing argument and fails)
// See also: contains, intersect
//
// CORA: contSet/@zonotope/isIntersecting_.m
// CORA authors: Niklas Kochdumper
// Authors:       Tobias Ladner, Claude Sonnet 5.5, Claude Opus 5.5
// Written:       03-November-2026
// Last update:   04-November-2026
// Reviewed:      ---

#include "cont_set/zonotope/private/priv.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/tensor/tolerance.h"
#include "cora/global/within_tol.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether the boxes around Z and S, S enlarged by tol, overlap in every coordinate.
bool aux_boxes_overlap(const Zonotope &Z, const Interval &S, double tol) {
    const Interval box = Z.interval();
    const Tensor apart = box.inf().subtract(S.sup() + tol).maximum(S.inf() - tol - box.sup());
    return apart.max().item() <= 0;
}

/// Whether c1 + G1 a = c2 + G2 b + e, |a|, |b| <= 1, |e| <= tol has a solution: the point c2 - c1
/// is in the zonotope with the generators [G1 G2 tol I], a linear program.
bool aux_overlap_exact(const Zonotope &Z, const Zonotope &S, double tol) {
    const Tensor slack = Z.center().eye_like(Z.dim()) * tol;
    const Zonotope sum(Z.center() * 0.0,
                       Tensor::concat({Z.generators(), S.generators(), slack}, -1));
    const double scaling =
        priv_zonotope_norm(priv_host_zonotope(sum), (S.center() - Z.center()).data());
    return scaling <= 1 || within_tol(scaling, 1, default_tol(Z.center().dtype()));
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool Zonotope::is_intersecting(const Zonotope &S, const std::string &type, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, c_.dtype(), "is_intersecting");
    // two zonotopes: a program on the sum, or the boxes around them
    CORA_CHECK(S.dim() == dim(), Error::dimension_mismatch, [&] {
        return "is_intersecting: the zonotopes differ in dimension, " + std::to_string(dim()) +
               " and " + std::to_string(S.dim());
    });
    const Refusal refusal = priv_intersecting_refusal(type, c_, S.center());
    CORA_CHECK(!refusal, refusal->first, [&] { return "is_intersecting: " + refusal->second; });
    return type == "exact" ? aux_overlap_exact(*this, S, tol)
                           : aux_boxes_overlap(*this, S.interval(), tol);
}

bool Zonotope::is_intersecting(const Interval &I, const std::string &type, Tol tol_arg) const {
    detail::Entry entry;
    // a box is the zonotope of its axes
    CORA_CHECK(I.dim() == dim() && I.inf().shape().back() == 1, Error::dimension_mismatch, [&] {
        return "is_intersecting: the interval must be a box of the dimension " +
               std::to_string(dim()) + " of the zonotope, got bounds of the shape " +
               shape_text(I.inf().shape());
    });
    const double tol = resolve_tol(entry, tol_arg, c_.dtype(), "is_intersecting");
    const Refusal refusal = priv_intersecting_refusal(type, c_, I.inf());
    CORA_CHECK(!refusal, refusal->first, [&] { return "is_intersecting: " + refusal->second; });
    if (I.is_empty()) return false;
    return type == "exact" ? aux_overlap_exact(*this, I.zonotope(), tol)
                           : aux_boxes_overlap(*this, I, tol);
}

bool Zonotope::is_intersecting(const Tensor &p, const std::string &type, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, c_.dtype(), "is_intersecting");
    // points are in the set or not: both types decide by containment
    const Refusal refusal = priv_intersecting_points_refusal(type, c_, p);
    CORA_CHECK(!refusal, refusal->first, [&] { return "is_intersecting: " + refusal->second; });
    return contains(p, "exact", tol);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
