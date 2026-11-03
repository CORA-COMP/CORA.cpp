// intersect - the intersection of a constrained zonotope with a set, as CORA's and_
//
// Syntax:   cZres = cz.intersect(S);
// Inputs:   S - a constrained zonotope, a zonotope or a box of the same dimension
// Outputs:  cZres - exact: the factors of both sets stay, each with its constraints, and one
//           equality per dimension makes the points agree, c + G x = c2 + G2 x2; without the
//           zero rows and generators (compact)
// See also: con_intersect, compact, Zonotope::intersect
//
// CORA: contSet/@conZonotope/and_.m

#include "cont_set/con_zonotope/private/priv.h"
#include "global/format.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Why the intersection is refused, if it is: the sets must have the same dimension.
detail::Refusal aux_refusal(const ConZonotope &cz, const ConZonotope &S) {
    if (S.dim() == cz.dim()) return std::nullopt;
    return std::pair{Error::dimension_mismatch, "intersect: the sets differ in dimension, " +
                                                    std::to_string(cz.dim()) + " and " +
                                                    std::to_string(S.dim())};
}

/// The intersection of two sets: the agreement rows of the images c + G x, without zeros.
ConZonotope aux_intersection(const ConZonotope &cz, const ConZonotope &S) {
    return priv_compact(priv_agreement(cz, S, cz.c(), cz.generators()));
}

} // namespace


// ===========================================  MAIN  =========================================== //

ConZonotope ConZonotope::intersect(const ConZonotope &S) const {
    detail::Entry entry;
    const detail::Refusal refusal = aux_refusal(*this, S);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_intersection(*this, S);
}

ConZonotope ConZonotope::intersect(const Zonotope &Z) const {
    detail::Entry entry;
    const ConZonotope S(Z);
    const detail::Refusal refusal = aux_refusal(*this, S);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_intersection(*this, S);
}

ConZonotope ConZonotope::intersect(const Interval &I) const {
    detail::Entry entry;
    const ConZonotope S(I);
    const detail::Refusal refusal = aux_refusal(*this, S);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_intersection(*this, S);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
