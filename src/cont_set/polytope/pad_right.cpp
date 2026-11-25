// pad_right - the polytope with new free coordinates before the old, as CORALean's padRight
//
// Every row gets k zero columns on the left, the offsets stay: R^k times the set. A polytope in
// vertex form is converted by constraints first (not a batch). The empty-set flag is kept.
//
// Syntax:   Pres = P.pad_right(k);
// Inputs:   P - polytope of dimension n
//           k - the number of new coordinates, k >= 0
// Outputs:  Pres - {(y, x) | y in R^k, x in P}, in halfspace form
// See also: pad_left, stack_rows, Polytope::lift
//
// CORA.cpp only: CORALean's HPolytope.padRight, the factor rows of a constrained zonotope
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The rows M (..., rows, n) with k zero columns before them.
Tensor aux_padded(const Tensor &M, int64_t k) {
    return Tensor::concat({zeros_block(M, M.shape()[M.ndim() - 2], k), M}, -1);
}

} // namespace


// ===========================================  MAIN  =========================================== //

Polytope Polytope::pad_right(int64_t k) const {
    detail::Entry entry;
    CORA_CHECK(k >= 0, Error::wrong_value, [&] {
        return "pad_right: the number of new coordinates must be nonnegative, got " +
               std::to_string(k);
    });
    const Constraints C = constraints();
    return Polytope(Parts{aux_padded(C.A, k), C.b, aux_padded(C.Ae, k), C.be, Tensor(), empty_});
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
