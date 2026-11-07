// zonotope - a zonotope that encloses a bounded polytope, as CORA's polytope.zonotope
//
// "outer" (default) is the parallelotope along the axes of an ellipsoid E = (c, Q): with R =
// sqrt(Q) the box of R^+ (P - c) is exact from the vertices, and the zonotope is c + R box. CORA
// takes the largest inscribed ellipsoid, a semidefinite program; here E is the covariance ellipsoid
// of the vertices (docs/cora-differences.md); every E gives an enclosure. "inner" (a zonotope
// inside) and "exact" are Error::not_supported, as "exact" is in CORA. R^n and unbounded sets are
// Error::special_error; the empty set has no zonotope. Not a batch.
//
// Syntax:   Zonotope Z = P.zonotope();   Zonotope Z = P.zonotope("outer");
// Inputs:   method - "outer" (default), "inner" or "exact"
// Outputs:  Z - a zonotope that contains the set
// See also: Polytope::zono_bundle, Polytope::ellipsoid, Polytope::interval
//
// CORA: contSet/@polytope/zonotope.m
// CORA authors: Mark Wetzlinger, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The square root of a symmetric positive semidefinite matrix, by its eigendecomposition.
Tensor aux_sqrtm(const Tensor &Q) {
    const auto [values, vectors] = Q.eigh();
    const Tensor roots = values.maximum(0.0).sqrt();
    return vectors.multiply(roots.matrix_transpose()).matmul(vectors.matrix_transpose());
}

} // namespace


// ===========================================  MAIN  =========================================== //

Zonotope Polytope::zonotope(const std::string &method) const {
    detail::Entry entry;
    CORA_CHECK(
        method == "outer" || method == "inner" || method == "exact", Error::wrong_value,
        [&] { return "zonotope: the method must be exact, outer or inner, got '" + method + "'"; });
    const Tensor &like = is_h_rep() ? A_ : V_;
    // the source must be a single, bounded set that is not empty
    priv_require_enclosable(*this, "zonotope", "a zonotope");
    CORA_CHECK(!priv_polytope_is_empty(*this), Error::special_error,
               [] { return std::string("The empty set is no zonotope."); });
    CORA_CHECK(method == "outer", Error::not_supported,
               [&] { return "zonotope: the method '" + method + "' is not supported"; });
    const Tensor V = priv_points_tensor(like, dim(), priv_polytope_vertices(*this));
    // the axes of the covariance ellipsoid, and the box of the vertices in them
    const Ellipsoid E = Ellipsoid::enclose_points(V);
    const Tensor R = aux_sqrtm(E.Q());
    const Tensor Y = R.pinv().matmul(V - E.q());
    const Tensor lower = Y.min(-1, true), upper = Y.max(-1, true);
    const Tensor center = (upper + lower) * 0.5, radius = (upper - lower) * 0.5;
    return {E.q() + R.matmul(center), R.multiply(radius.matrix_transpose())};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
