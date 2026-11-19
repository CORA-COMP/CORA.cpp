// ellipsoid_norm - the norm whose unit ball is an ellipsoid, as CORA's ellipsoidNorm
//
// It is computed from Q \ p for a full-rank Q; else from the singular value decomposition Q = U S
// V', with singular values and entries of U' p below 1e-10 set to zero, so that a direction
// outside the range of Q has norm Inf.
//
// Syntax:   Tensor res = E.ellipsoid_norm(p);
// Inputs:   p - vectors (..., n), the batch of E broadcast
// Outputs:  res - sqrt(|p' Q^+ p|) (...)
// See also: generators, contains
//
// CORA: contSet/@ellipsoid/ellipsoidNorm.m
// CORA authors: Adrian Kulmburg
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5
// Written:       04-November-2026
// Last update:   19-November-2026
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

// the rows are the columns of the kernel (CORA's layout), the result its 1 x 1 matrices
Tensor Ellipsoid::ellipsoid_norm(const Tensor &p) const {
    detail::Entry entry;
    CORA_CHECK(!priv_is_empty(*this), Error::empty_set,
               [] { return std::string("ellipsoid_norm: the ellipsoid is empty"); });
    CORA_CHECK(p.defined() && is_vector(p, dim()), Error::wrong_value, [&] {
        return "ellipsoid_norm: p must be a vector (..., " + std::to_string(dim()) +
               "), got the shape " + (p.defined() ? shape_text(p.shape()) : "()");
    });
    return priv_ellipsoid_norm(*this, p.expand_dims(-1)).squeeze(-1).squeeze(-1);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
