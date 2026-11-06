// plus - the Minkowski sum of an ellipsoid with a vector or an ellipsoid, as CORA's ellipsoid.plus
//
// A vector shifts the center. Two ellipsoids give the enclosure of Halder (2018), (1 + 1/b) Q1 +
// (1 + b) Q2 for the fixed point b of the eigenvalues of Q1^-1 Q2; points and one dimension are
// exact. The other modes of CORA need an SDP solver, the sums with boxes and zonotopes the minimal
// enclosing ellipsoid, and the directions L the intersection of ellipsoids: Error::not_supported.
//
// Syntax:   Eres = E.plus(v);   Eres = E.plus(S);   Eres = E.plus(S, mode);
// Inputs:   v - column (n, 1)
//           S - ellipsoid
//           mode - "outer:halder" (the default of CORA for two ellipsoids)
// Outputs:  Eres - an ellipsoid that holds E + S
// Gradients: the sum with a vector is differentiable; the weight of Halder is a host number
// See also: mink_diff, Zonotope::plus
//
// CORA: contSet/@ellipsoid/plus.m
// CORA authors: Victor Gassmann, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "cora/global/within_tol.h"
#include "global/format.h"

#include <cmath>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Ellipsoid Ellipsoid::plus(const Tensor &v) const {
    detail::Entry entry;
    CORA_CHECK(is_column(v, dim()), Error::dimension_mismatch, [&] {
        return "plus: the vector must be a column of the dimension " + std::to_string(dim()) +
               ", got the shape " + shape_text(v.shape());
    });
    // the sum with an empty set is empty
    if (priv_is_empty(*this)) return *this;
    return {Q_, q_ + v, tol_};
}

Ellipsoid Ellipsoid::plus(const Ellipsoid &S, const std::string &mode) const {
    detail::Entry entry;
    CORA_CHECK(
        mode == "outer" || mode == "outer:halder" || mode == "inner", Error::wrong_value, [&] {
            return "plus: the mode must be 'outer', 'outer:halder' or 'inner', got '" + mode + "'";
        });
    CORA_CHECK(S.dim() == dim(), Error::dimension_mismatch, [&] {
        return "plus: the ellipsoids differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(S.dim());
    });
    if (priv_is_empty(*this) || priv_is_empty(S)) return Ellipsoid::empty(dim());
    CORA_CHECK(Q_.ndim() == 2 && S.Q_.ndim() == 2, Error::not_supported,
               [] { return std::string("plus: it supports single ellipsoids, not a batch"); });
    // points add exactly: the sum with a point is a shift, with the origin the ellipsoid itself
    if (priv_is_point(S)) return {Q_, q_ + S.q_, tol_};
    if (priv_is_point(*this)) return {S.Q_, q_ + S.q_, S.tol_};
    // in one dimension the sum of the intervals is the sum of the radii
    if (dim() == 1) {
        const Tensor radius = Q_.sqrt() + S.Q_.sqrt();
        return {radius.square(), q_ + S.q_, std::min(tol_, S.tol_)};
    }
    CORA_CHECK(mode == "outer:halder", Error::not_supported, [&] {
        return "plus: the mode '" + mode + "' needs an SDP solver; use 'outer:halder'";
    });
    CORA_CHECK(is_full_dim() && S.is_full_dim(), Error::not_supported, [] {
        return std::string("plus: the enclosure of two degenerate ellipsoids is not supported");
    });
    const double tol = std::min(tol_, S.tol_);
    // the eigenvalues l of Q1^-1 Q2, from the simultaneous diagonalization
    const std::vector<double> l = priv_simdiag(Q_, S.Q_, tol).second.diagonal().data();
    double beta = 0.5, next = beta;
    for (int step = 0;; ++step) {
        double numerator = 0.0, denominator = 0.0;
        for (const double li : l) {
            numerator += 1.0 / (1.0 + beta * li);
            denominator += li / (1.0 + beta * li);
        }
        next = std::sqrt(numerator / denominator);
        if (within_tol(beta, next, tol)) break;
        CORA_CHECK(step < numerics::kFixedPointSteps, Error::not_supported,
                   [] { return std::string("plus: the fixed point of Halder did not converge"); });
        beta = next;
    }
    CORA_CHECK(next > 0.0 && !within_tol(next, 0.0, tol), Error::special_error,
               [] { return std::string("plus: the weight of Halder is almost zero"); });
    return {Q_ * (1.0 + 1.0 / next) + S.Q_ * (1.0 + next), q_ + S.q_, tol};
}

Set Ellipsoid::translate_impl(const Tensor &v) const { return plus(v); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
