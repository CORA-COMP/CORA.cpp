// isequal - whether two ellipsoids are equal, as CORA's ellipsoid.isequal
//
// The shape matrices and the centers agree entrywise up to tol, or both are empty; ellipsoids of
// another dimension differ. For a batch, whether every member agrees.
//
// Syntax:   res = E.isequal(S);   res = E.isequal(S, tol);
// Inputs:   S - ellipsoid
//           tol - default: the smaller tolerance of the two ellipsoids
// Outputs:  res - whether they are equal
// See also: is_bigger, Capsule::isequal
//
// CORA: contSet/@ellipsoid/isequal.m
// CORA authors: Victor Gassmann
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/ellipsoid/private/priv.h"
#include "cora/cont_set/ellipsoid/ellipsoid.h"
#include "cora/global/within_tol.h"
#include "global/format.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool Ellipsoid::isequal(const Ellipsoid &S, Tol tol_arg) const {
    detail::Entry entry;
    const double tol =
        resolve_tol(entry, tol_arg ? tol_arg : Tol(std::min(tol_, S.tol_)), Q_.dtype(), "isequal");
    if (dim() != S.dim()) return false;
    // empty and nonempty differ; two empty ellipsoids are equal
    if (priv_is_empty(*this) || priv_is_empty(S)) return priv_is_empty(*this) && priv_is_empty(S);
    if (Q_.shape() != S.Q_.shape()) return false;
    return within_tol(Q_, S.Q_, tol).min().item() != 0.0 &&
           within_tol(q_, S.q_, tol).min().item() != 0.0;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
