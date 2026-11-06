// is_intersecting - whether a constrained zonotope and a set or points intersect
//
// Syntax:   res = cz.is_intersecting(S);   res = cz.is_intersecting(S, "approx");
//           res = cz.is_intersecting(points, "exact", tol);
// Inputs:   S - a constrained zonotope, a zonotope, a box, or points (n, N);  type - "exact"
//           (default) or "approx";  tol - for points, the tolerance of contains; the sets are
//           compared without it (as CORA); default: the repository tolerance
// Outputs:  res - "exact": the intersection is not empty (a linear program); "approx": the boxes
//           around the two sets overlap, so it can say true for disjoint sets. For points all of
//           them must lie in the set (CORA's call to contains_ has a missing argument and fails
//           for a constrained zonotope too). A box without points intersects nothing. Not a batch
// See also: intersect, contains
//
// CORA: contSet/@conZonotope/isIntersecting_.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "cont_set/zonotope/private/priv.h"
#include "cora/global/tensor/tolerance.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Whether the boxes of two sets overlap in every coordinate (an empty box overlaps nothing).
bool aux_boxes_overlap(const Interval &a, const Interval &b) {
    if (a.is_empty() || b.is_empty()) return false;
    return a.inf().subtract(b.sup()).maximum(b.inf().subtract(a.sup())).max().item() <= 0;
}

/// Whether a constrained zonotope and a set meet, by the type (the checks are the caller's).
bool aux_meets(const ConZonotope &cz, const ConZonotope &S, const std::string &type) {
    // the exact test: the intersection has a point; the approximate one: the boxes overlap
    if (type == "approx") return aux_boxes_overlap(cz.interval(), S.interval());
    return !priv_is_empty(priv_host_con_zonotope(cz.intersect(S)));
}

/// Why two sets are refused: the type and batch, then the dimension.
detail::Refusal aux_refusal(const ConZonotope &cz, const ConZonotope &S, const std::string &type) {
    if (const Refusal r = priv_intersecting_refusal(type, cz.c(), S.c())) return r;
    return priv_pair_refusal("is_intersecting", cz, S, true);
}

} // namespace


// ===========================================  MAIN  =========================================== //

bool ConZonotope::is_intersecting(const ConZonotope &S, const std::string &type, Tol tol) const {
    detail::Entry entry;
    // CORA compares the sets without a tolerance; it is only checked
    static_cast<void>(resolve_tol(entry, tol, c_.dtype(), "is_intersecting"));
    const detail::Refusal refusal = aux_refusal(*this, S, type);
    CORA_CHECK(!refusal, refusal->first, [&] { return "is_intersecting: " + refusal->second; });
    return aux_meets(*this, S, type);
}

// a zonotope is a set without constraints
bool ConZonotope::is_intersecting(const Zonotope &Z, const std::string &type, Tol tol) const {
    detail::Entry entry;
    static_cast<void>(resolve_tol(entry, tol, c_.dtype(), "is_intersecting"));
    const ConZonotope S(Z);
    const detail::Refusal refusal = aux_refusal(*this, S, type);
    CORA_CHECK(!refusal, refusal->first, [&] { return "is_intersecting: " + refusal->second; });
    return aux_meets(*this, S, type);
}

// a box is a set without constraints; an empty box meets nothing
bool ConZonotope::is_intersecting(const Interval &I, const std::string &type, Tol tol) const {
    detail::Entry entry;
    static_cast<void>(resolve_tol(entry, tol, c_.dtype(), "is_intersecting"));
    CORA_CHECK(I.inf().shape().back() == 1, Error::dimension_mismatch, [&] {
        return "is_intersecting: the interval must be a box with column bounds, got the shape " +
               shape_text(I.inf().shape());
    });
    if (I.is_empty()) return false;
    const ConZonotope S(I);
    const detail::Refusal refusal = aux_refusal(*this, S, type);
    CORA_CHECK(!refusal, refusal->first, [&] { return "is_intersecting: " + refusal->second; });
    return aux_meets(*this, S, type);
}

bool ConZonotope::is_intersecting(const Tensor &p, const std::string &type, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, c_.dtype(), "is_intersecting");
    const Refusal refusal = priv_intersecting_points_refusal(type, c_, p);
    CORA_CHECK(!refusal, refusal->first, [&] { return "is_intersecting: " + refusal->second; });
    // points are in the set or not: both types decide by containment
    return contains(p, "exact", tol);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
