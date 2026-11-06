// isequal - whether two capsules are the same set
//
// Syntax:   res = C.isequal(S);   res = C.isequal(S, tol);
// Inputs:   S - capsule
//           tol - the centers, generators and radii agree as within_tol; default:
//           the repository tolerance (docs/architecture.md)
// Outputs:  res - true if both are empty, or neither is and the centers, generators and radii
//           agree up to tol; a different dimension gives false. CORA compares with a strict <,
//           so a tolerance of 0 never holds there. Of a batch: every member
// See also: contains, Interval::isequal
//
// CORA: contSet/@capsule/isequal.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/capsule/private/priv.h"
#include "cora/cont_set/capsule/capsule.h"
#include "cora/global/within_tol.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool Capsule::isequal(const Capsule &S, Tol tol_arg) const {
    detail::Entry entry;
    const double tol = resolve_tol(entry, tol_arg, c_.dtype(), "isequal");
    if (S.dim() != dim()) return false;
    const auto agree = [&](const Tensor &a, const Tensor &b) {
        return within_tol(a, b, tol).min(-2, true);
    };
    const Tensor same = agree(c_, S.c_).multiply(agree(g_, S.g_)).multiply(agree(r_, S.r_));
    const Tensor mine = priv_empty_mask(r_), theirs = priv_empty_mask(S.r_);
    // two empty capsules are equal whatever their other fields hold
    const Tensor both_empty = mine.multiply(theirs);
    const Tensor both_set = (1.0 - mine).multiply(1.0 - theirs).multiply(same);
    return priv_all(both_empty.maximum(both_set));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
