// is_full_dim - whether an ellipsoid has a nonempty interior, as CORA's isFullDim
//
// Syntax:   res = E.is_full_dim();   res = E.is_full_dim(tol);
// Inputs:   tol - singular values of Q up to tol times the largest count as zero; default: the
//           tolerance of the ellipsoid (CORA's TOL)
// Outputs:  res - true if the rank of Q is n; false for an empty ellipsoid. For a batch, true if
//           every member is
// See also: is_bounded, generators
//
// CORA: contSet/@ellipsoid/isFullDim.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool Ellipsoid::is_full_dim(Tol tol) const {
    detail::Entry entry;
    const double t = tol ? resolve_tol(entry, tol, Q_.dtype(), "is_full_dim") : tol_;
    if (priv_is_empty(*this)) return false;
    const Tensor full = priv_rank(Q_, t).equal(Q_.ones_like({1, 1}) * double(dim()));
    return full.reshape({full.numel()}).all(0).item() != 0.0;
}

bool Ellipsoid::is_full_dim_impl(Tol tol) const { return is_full_dim(tol); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
