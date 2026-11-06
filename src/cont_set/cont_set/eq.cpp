// eq - whether two sets are equal, as CORA's contSet.eq
//
// Syntax:   res = eq(S1, S2);   res = eq(S1, S2, tol);   res = (S1 == S2);
// Inputs:   S1, S2 - sets of one dimension
//           tol - as isequal; default: the repository tolerance
// Outputs:  res - isequal(S1, S2, tol)
// See also: isequal, ne
//
// CORA: contSet/@contSet/eq.m
// CORA authors: Mark Wetzlinger, Tobias Ladner
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       03-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cora/cont_set/registry.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

bool eq(const ContSet &S1, const ContSet &S2, Tol tol) {
    detail::Entry entry;
    const detail::Refusal r = detail::refusal(BinaryOp::isequal, S1, S2, tol);
    CORA_CHECK(!r, r->first, [&] { return r->second; });
    return detail::lookup(BinaryOp::isequal, S1, S2, tol);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
