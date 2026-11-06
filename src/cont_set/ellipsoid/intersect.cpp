// intersect - the intersection of an ellipsoid with a point or an ellipsoid, as CORA's and_
//
// The ellipsoids are single, not batched. A point is contained up to the machine epsilon. The
// same when one of the ellipsoids is a point (also the second, which CORA's solver path handles),
// and the empty ellipsoid when one is empty. The intersection of two ellipsoids that are no
// points (the SDP-based outer and inner approximations, and the polytope and hyperplane cases) is
// Error::not_supported.
//
// Syntax:   Ellipsoid R = E.intersect(p);   Ellipsoid R = E.intersect(S, mode);
// Inputs:   p - a point (n, 1)
//           S - an ellipsoid
//           mode - "outer" (default) or "inner"
// Outputs:  R - the point as an ellipsoid (Q = 0) if the ellipsoid contains it, else the empty one
// See also: contains, Zonotope::intersect
//
// CORA: contSet/@ellipsoid/and_.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The point p as the result of an intersection that holds it: Q = 0 around p.
Ellipsoid aux_point(const Ellipsoid &like, const Tensor &p) {
    const int64_t n = like.dim();
    return {like.Q().zeros_like({n, n}), p};
}

} // namespace


// ===========================================  MAIN  =========================================== //

Ellipsoid Ellipsoid::intersect(const Tensor &p) const {
    detail::Entry entry;
    CORA_CHECK(p.defined() && is_column(p, dim()) && p.ndim() == 2, Error::dimension_mismatch, [&] {
        return "intersect: the point must be a column (" + std::to_string(dim()) + ", 1)";
    });
    const detail::Refusal refusal = priv_unbatched_refusal(*this, "intersect");
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    // the machine epsilon is CORA's tolerance of this test
    if (contains(p, "exact", machine_eps(Q_.dtype()))) return aux_point(*this, p);
    return empty(dim());
}

Ellipsoid Ellipsoid::intersect(const Ellipsoid &S, const std::string &mode) const {
    detail::Entry entry;
    CORA_CHECK(S.dim() == dim(), Error::dimension_mismatch, [&] {
        return "intersect: the ellipsoids differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(S.dim());
    });
    CORA_CHECK(mode == "outer" || mode == "inner", Error::wrong_value, [&] {
        return "intersect: the mode must be 'outer' or 'inner', got '" + mode + "'";
    });
    detail::Refusal refusal = priv_unbatched_refusal(*this, "intersect");
    if (!refusal) refusal = priv_unbatched_refusal(S, "intersect");
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    if (priv_is_empty(*this) || priv_is_empty(S)) return empty(dim());
    const double eps = machine_eps(Q_.dtype());
    // a point intersects an ellipsoid in itself or in nothing
    if (priv_is_point(*this))
        return S.contains(q_, "exact", eps) ? aux_point(*this, q_) : empty(dim());
    if (priv_is_point(S)) return contains(S.q(), "exact", eps) ? aux_point(S, S.q()) : empty(dim());
    // the other cases need a solver
    CORA_FAIL(Error::not_supported,
              "intersect: two ellipsoids that are no points are intersected by SDP solvers, which "
              "are not available");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
