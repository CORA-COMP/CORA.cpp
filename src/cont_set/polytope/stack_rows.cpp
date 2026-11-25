// stack_rows - the rows of two polytopes stacked, as CORALean's stackRows
//
// The inequalities of P, then those of P2, and the equalities likewise: the intersection, with no
// row dropped or merged (intersect returns P itself when P2 is R^n). A polytope in vertex form is
// converted by constraints first (not a batch). Empty where either set is flagged empty.
//
// Syntax:   Pres = P.stack_rows(P2);
// Inputs:   P, P2 - polytopes of one dimension and batch shape
// Outputs:  Pres - the intersection of P and P2, in halfspace form
// See also: intersect, pad_left, pad_right
//
// CORA.cpp only: CORALean's HPolytope.stackRows, the factor rows of a constrained zonotope
// Authors:       Claude Opus 5.5
// Written:       25-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/polytope/private/priv.h"
#include "global/format.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Why the rows do not stack, if they do not: one dimension and one batch shape.
detail::Refusal aux_refusal(const Polytope::Constraints &C1, const Polytope::Constraints &C2) {
    const Shape a = C1.A.shape(), b = C2.A.shape();
    if (a.back() != b.back())
        return std::pair{Error::dimension_mismatch, "stack_rows: the sets differ in dimension, " +
                                                        std::to_string(a.back()) + " and " +
                                                        std::to_string(b.back())};
    if (a.size() != b.size() || !std::equal(a.begin(), a.end() - 2, b.begin()))
        return std::pair{Error::dimension_mismatch,
                         "stack_rows: the sets must have the same batch shape, got " +
                             shape_text(a) + " and " + shape_text(b)};
    return std::nullopt;
}

} // namespace


// ===========================================  MAIN  =========================================== //

Polytope Polytope::stack_rows(const Polytope &P2) const {
    detail::Entry entry;
    const Constraints C1 = constraints(), C2 = P2.constraints();
    const detail::Refusal refusal = entry.check() ? aux_refusal(C1, C2) : std::nullopt;
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return Polytope(Parts{Tensor::concat({C1.A, C2.A}, -2), Tensor::concat({C1.b, C2.b}, -1),
                          Tensor::concat({C1.Ae, C2.Ae}, -2), Tensor::concat({C1.be, C2.be}, -1),
                          Tensor(), empty_.either(P2.empty_mask())});
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
