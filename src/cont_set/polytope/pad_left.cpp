// pad_left - the polytope with new free coordinates after the old, as CORALean's padLeft
//
// Every row gets k zero columns on the right, the offsets stay: the set times R^k. A polytope in
// vertex form is converted by constraints first (not a batch). The empty-set flag is kept.
//
// Syntax:   Pres = P.pad_left(k);
// Inputs:   P - polytope of dimension n
//           k - the number of new coordinates, k >= 0
// Outputs:  Pres - {(x, y) | x in P, y in R^k}, in halfspace form
// See also: pad_right, stack_rows, Polytope::lift
//
// CORA.cpp only: CORALean's HPolytope.padLeft, the factor rows of a constrained zonotope
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The rows M (..., rows, n) with k zero columns after them.
Tensor aux_padded(const Tensor &M, int64_t k) {
    return Tensor::concat({M, zeros_block(M, M.shape()[M.ndim() - 2], k)}, -1);
}

} // namespace


// ===========================================  MAIN  =========================================== //

Polytope Polytope::pad_left(int64_t k) const {
    detail::Entry entry;
    CORA_CHECK(k >= 0, Error::wrong_value, [&] {
        return "pad_left: the number of new coordinates must be nonnegative, got " +
               std::to_string(k);
    });
    const Constraints C = constraints();
    return Polytope(Parts{aux_padded(C.A, k), C.b, aux_padded(C.Ae, k), C.be, Tensor(), empty_});
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
