// priv - the private functions of ellipsoid, as CORA's @ellipsoid/private and its helpers
//
// Not for callers of the class: the operations of the class use them.
//
// Syntax:   bool empty = priv_is_empty(E);   Tensor r = priv_rank(E.Q(), E.tol());
//           auto [T, D] = priv_simdiag(Q1, Q2, tol);
// See also: ellipsoid.h
//
// Authors:       Claude Sonnet 5.5
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#pragma once

#include "cora/cont_set/ellipsoid/ellipsoid.h"

#include <utility>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// Predicates --------------------------------------------------------------------------------------

/// Whether E is the empty ellipsoid: its shape matrix has no entries.
bool priv_is_empty(const Ellipsoid &E);

/// Whether the shape matrix of the unbatched E is zero up to eps: E is a point (CORA's
/// representsa_(E, 'point', eps)).
bool priv_is_point(const Ellipsoid &E);

/// Why two ellipsoids are refused by an operation that decides for single, nonempty ones: unequal
/// dimensions, a batch, an empty ellipsoid (the identifier and text), if any.
detail::Refusal priv_pair_refusal(const Ellipsoid &E1, const Ellipsoid &E2, const char *operation);

/// Why a batch is refused by an operation that decides per ellipsoid (the identifier and text of
/// the Error::not_supported), if Q has batch axes.
detail::Refusal priv_unbatched_refusal(const Ellipsoid &E, const char *operation);

// Linear algebra ----------------------------------------------------------------------------------

/// The rank of the symmetric matrices Q (..., n, n) as CORA's rank(E): the singular values above
/// tol times the largest; as a count (..., 1, 1).
Tensor priv_rank(const Tensor &Q, double tol);

/// The simultaneous diagonalization of two symmetric matrices, the first positive definite (CORA's
/// simdiag): T with T M1 T' = I and D = T M2 T' diagonal. Error::special_error unless the
/// matrices are symmetric up to tol, M1 has eigenvalues at least tol and M2 at least -tol.
std::pair<Tensor, Tensor> priv_simdiag(const Tensor &M1, const Tensor &M2, double tol);

/// The shape matrix M Q M' of the image of an ellipsoid under M, symmetrized (CORA's mtimes).
Tensor priv_congruence(const Tensor &M, const Tensor &Q);

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
