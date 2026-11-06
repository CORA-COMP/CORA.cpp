// plus - the Minkowski sum of a constrained zonotope with a set or a point, as CORA's plus
//
// The constraints sit on the block diagonal; a point moves the center. One batch shape.
//
// Syntax:   cZres = cz.plus(S);   cZres = cz.plus(v);   cZres = cz + S;
// Inputs:   S - a constrained zonotope, a zonotope or a box of the dimension of cz
//           v - a column (..., n, 1)
// Outputs:  cZres - exact: the centers add, the generators of both join
// See also: linear_map, Zonotope::plus
//
// CORA: contSet/@conZonotope/plus.m
// CORA authors: Dmitry Grebenyuk, Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// The sum of two sets after the checks of their batch shape and dimension.
ConZonotope aux_sum(const ConZonotope &cz, const ConZonotope &S) {
    detail::Entry entry;
    const detail::Refusal refusal = priv_pair_refusal("plus", cz, S, true);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return priv_minkowski_sum(cz, S);
}

} // namespace


// ===========================================  MAIN  =========================================== //

ConZonotope ConZonotope::plus(const ConZonotope &S) const { return aux_sum(*this, S); }

ConZonotope ConZonotope::plus(const Zonotope &Z) const { return aux_sum(*this, ConZonotope(Z)); }

ConZonotope ConZonotope::plus(const Interval &I) const { return aux_sum(*this, ConZonotope(I)); }

ConZonotope ConZonotope::plus(const Tensor &v) const {
    detail::Entry entry;
    CORA_CHECK(v.ndim() >= 2 && v.shape().back() == 1 && v.shape()[v.ndim() - 2] == dim(),
               Error::dimension_mismatch, [&] {
                   return "plus: the point must be a column of " + std::to_string(dim()) +
                          " entries, got the shape " + shape_text(v.shape());
               });
    return {c_ + v, G_, A_, b_};
}

Set ConZonotope::translate_impl(const Tensor &v) const { return plus(v); }

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
