// unite - an ellipsoid that holds the union of two, as CORA's ellipsoid.or
//
// Mode "inner" returns the ellipsoid itself, as CORA. "outer": an empty operand gives the other,
// and a point inside the other ellipsoid is absorbed. The smallest enclosing ellipsoid of two
// proper ones is an SDP, which CORA.cpp lacks: Error::not_supported.
//
// Syntax:   Eres = E.unite(S);   Eres = E.unite(S, mode);   Eres = E.unite(p, mode);
// Inputs:   S - ellipsoid
//           p - a point (n, 1)
//           mode - "outer" (default) or "inner"
// Outputs:  Eres - the ellipsoid
// See also: convex_hull, contains
//
// CORA: contSet/@ellipsoid/or.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Ellipsoid Ellipsoid::unite(const Ellipsoid &S, const std::string &mode) const {
    detail::Entry entry;
    CORA_CHECK(mode == "outer" || mode == "inner", Error::wrong_value,
               [&] { return "unite: the mode must be 'outer' or 'inner', got '" + mode + "'"; });
    CORA_CHECK(S.dim() == dim(), Error::dimension_mismatch, [&] {
        return "unite: the ellipsoids differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(S.dim());
    });
    if (priv_is_empty(S)) return *this;
    if (mode == "inner" || priv_is_empty(*this)) return mode == "inner" ? *this : S;
    CORA_CHECK(Q_.ndim() == 2 && S.Q_.ndim() == 2, Error::not_supported,
               [] { return std::string("unite: it supports single ellipsoids, not a batch"); });
    // a point inside the other ellipsoid adds nothing
    if (priv_is_point(S) && contains(S.q_)) return *this;
    if (priv_is_point(*this) && S.contains(q_)) return S;
    CORA_FAIL(Error::not_supported,
              "unite: the smallest ellipsoid around two ellipsoids needs an SDP solver");
}

Ellipsoid Ellipsoid::unite(const Tensor &p, const std::string &mode) const {
    detail::Entry entry;
    CORA_CHECK(is_column(p, dim()), Error::dimension_mismatch, [&] {
        return "unite: the point must be a column of the dimension " + std::to_string(dim()) +
               ", got the shape " + shape_text(p.shape());
    });
    return unite(Ellipsoid(Q_.zeros_like({dim(), dim()}), p, tol_), mode);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
