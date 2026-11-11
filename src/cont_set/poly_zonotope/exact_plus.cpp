// exact_plus - the sum of two polynomial zonotopes that share their factors, as CORA's exactPlus
//
// Equal identifiers are the same factor in both sets, so the dependency between the summands
// stays: the sum of x and -x is {0} if both are the same polynomial zonotope. The exponent
// matrices are extended to the union of the identifiers and the generators join.
//
// Syntax:   S = pZ.exact_plus(pZ2);
// Inputs:   pZ2 - a polynomial zonotope of the same dimension
// Outputs:  S - the sum, a subset of plus
// See also: plus
//
// CORA: contSet/@polyZonotope/exactPlus.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

PolyZonotope PolyZonotope::exact_plus(const PolyZonotope &S) const {
    detail::Entry entry;
    CORA_CHECK(S.dim() == dim(), Error::dimension_mismatch, [&] {
        return "exact_plus: the sets differ in dimension, " + std::to_string(dim()) + " and " +
               std::to_string(S.dim());
    });
    // both exponent matrices over the union of the identifiers
    const MergedIds ids = priv_merge_ids(id_, S.id_);
    Exponents E = priv_extend_exponents(E_, ids.position1, ids.id.size());
    const Exponents E2 = priv_extend_exponents(S.E_, ids.position2, ids.id.size());
    E.insert(E.end(), E2.begin(), E2.end());
    return {c_ + S.c_, Tensor::concat({G_, S.G_}, -1), Tensor::concat({GI_, S.GI_}, -1),
            std::move(E), ids.id};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
