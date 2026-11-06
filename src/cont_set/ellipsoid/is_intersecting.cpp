// is_intersecting - whether an ellipsoid meets points or an ellipsoid, as CORA's isIntersecting_
//
// Points: all of them lie in the ellipsoid (contains). Ellipsoid: an empty one meets nothing; if
// one is a point (up to tol) the other contains it; two proper ellipsoids need the distance of
// ellipsoids, a solver: Error::not_supported (CORA: noExactAlg).
//
// Syntax:   res = E.is_intersecting(p);   res = E.is_intersecting(S, tol);
// Inputs:   p - points (n, N)
//           S - ellipsoid
//           tol - default: the repository tolerance (docs/architecture.md)
// Outputs:  res - whether they meet
// See also: contains, distance
//
// CORA: contSet/@ellipsoid/isIntersecting_.m
// CORA authors: Victor Gassmann, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "cora/global/within_tol.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool Ellipsoid::is_intersecting(const Tensor &p, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, Q_.dtype(), "is_intersecting");
    CORA_CHECK(p.defined() && p.ndim() == 2 && p.shape()[0] == dim(), Error::dimension_mismatch,
               [&] {
                   return "is_intersecting: the points must be a matrix with " +
                          std::to_string(dim()) + " rows, got the shape " + shape_text(p.shape());
               });
    // points meet the ellipsoid if all of them lie in it
    return contains(p, "exact", tol);
}

bool Ellipsoid::is_intersecting(const Ellipsoid &S, Tol tol_arg) const {
    detail::Entry entry;
    CORA_CHECK(S.dim() == dim(), Error::dimension_mismatch, [&] {
        return "is_intersecting: the ellipsoids differ in dimension, " + std::to_string(dim()) +
               " and " + std::to_string(S.dim());
    });
    const double tol = resolve_tol(entry, tol_arg, Q_.dtype(), "is_intersecting");
    if (priv_is_empty(*this) || priv_is_empty(S)) return false;
    CORA_CHECK(Q_.ndim() == 2 && S.Q_.ndim() == 2, Error::not_supported, [] {
        return std::string("is_intersecting: it supports single ellipsoids, not a batch");
    });
    // a point meets the other ellipsoid if the other contains it
    const auto is_point = [&](const Ellipsoid &E) {
        return within_tol(E.Q_, E.Q_.zeros_like(), tol).min().item() != 0.0;
    };
    if (is_point(*this)) return S.contains(q_, "exact", tol);
    if (is_point(S)) return contains(S.q_, "exact", tol);
    CORA_FAIL(Error::not_supported,
              "is_intersecting: the distance of two ellipsoids needs a solver");
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
