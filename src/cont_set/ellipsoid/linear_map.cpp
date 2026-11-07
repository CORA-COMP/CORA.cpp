// linear_map - the image of an ellipsoid under a matrix, as CORA's ellipsoid.mtimes
//
// M Q M' (symmetrized) and M q: exact. The empty ellipsoid stays empty, with the rows of M as its
// dimension.
//
// Syntax:   Eres = E.linear_map(M);
// Inputs:   M - matrix (..., m, n)
// Outputs:  Eres - the ellipsoid M E
// See also: project, Zonotope::linear_map
//
// CORA: contSet/@ellipsoid/mtimes.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   07-November-2026
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Ellipsoid Ellipsoid::linear_map(const Tensor &M) const {
    detail::Entry entry;
    CORA_CHECK(M.defined() && M.ndim() >= 2 && M.size(-1) == dim(), Error::dimension_mismatch, [&] {
        return "linear_map: the matrix needs " + std::to_string(dim()) +
               " columns (the dimension of E), got the shape " + shape_text(M.shape());
    });
    if (priv_is_empty(*this)) return Ellipsoid::empty(M.size(-2));
    return {priv_congruence(M, Q_), M.matmul(q_), tol_};
}

Set Ellipsoid::linear_map_impl(const Tensor &M) const { return linear_map(M); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
