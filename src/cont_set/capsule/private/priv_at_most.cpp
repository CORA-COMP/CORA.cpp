// priv_at_most - where one number is at most another, up to a tolerance
//
// Syntax:   Tensor ok = priv_at_most(a, b, tol);
// Inputs:   a, b - tensors of one dtype, broadcast
//           tol - nonnegative
// Outputs:  ok - 1 where a <= b or the two agree up to tol (as within_tol), else 0: CORA's
//           `dist < r | withinTol(dist, r)`
// See also: priv_all, within_tol
//
// CORA: contSet/@capsule/contains_.m
// CORA authors: Niklas Kochdumper, Mark Wetzlinger
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/capsule/private/priv.h"
#include "cora/global/within_tol.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Tensor priv_at_most(const Tensor &a, const Tensor &b, double tol) {
    return a.less_equal(b).maximum(within_tol(a, b, tol));
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
