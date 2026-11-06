// split - a constrained zonotope cut into two that cover it, as CORA's conZonotope.split
//
// The pieces are the intersections of the set with the two halves of its box cut at the center of
// coordinate N; without N one pair per coordinate. They cover the set and are exact up to the
// linear programs of the box. An empty set is Error::empty_set. Not a batch.
//
// Syntax:   [cZ1, cZ2] = cz.split(N);   pieces = cz.split();
// Inputs:   N - a coordinate (0-based) of the box around the set
// Outputs:  pieces - the two pieces at the center of coordinate N (lower half first)
// See also: interval, intersect, Zonotope::split
//
// CORA: contSet/@conZonotope/split.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5, Claude Opus 5.5, Tobias Ladner
// Written:       04-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/con_zonotope/private/priv.h"
#include "cont_set/cont_set/private/priv_coordinate.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Why the split is refused, if it is: a batch, or an empty set.
detail::Refusal aux_refusal(const ConZonotope &cz) {
    if (cz.c().ndim() != 2)
        return std::pair{Error::not_supported,
                         std::string("split: it supports a single set, not a batch")};
    if (priv_is_empty(priv_host_con_zonotope(cz)))
        return std::pair{Error::empty_set, std::string("split: the set is empty")};
    return std::nullopt;
}

/// The two pieces of the box I cut at the center of coordinate d (CORA's aux_splitOneDim): the
/// center and radius are moved by half the radius, in the order CORA computes them.
std::pair<ConZonotope, ConZonotope> aux_split_dim(const ConZonotope &cz, const Interval &I,
                                                  int64_t d) {
    const Tensor c = I.center(), r = I.rad();
    std::vector<double> unit(std::size_t(cz.dim()), 0.0);
    unit[d] = 1.0;
    const Tensor shift = Tensor::like(c, unit, {cz.dim(), 1}).multiply(r * 0.5);
    const Tensor radius = r - shift;
    const Tensor lower = c - shift, upper = c + shift;
    return {cz.intersect(Interval(lower - radius, lower + radius)),
            cz.intersect(Interval(upper - radius, upper + radius))};
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::pair<ConZonotope, ConZonotope> ConZonotope::split(int64_t N) const {
    detail::Entry entry;
    detail::Refusal refusal = priv_coordinate_refusal(N, dim(), "split");
    if (!refusal) refusal = aux_refusal(*this);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_split_dim(*this, interval(), N);
}

std::vector<std::pair<ConZonotope, ConZonotope>> ConZonotope::split() const {
    detail::Entry entry;
    const detail::Refusal refusal = aux_refusal(*this);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    const Interval I = interval();
    std::vector<std::pair<ConZonotope, ConZonotope>> pieces;
    pieces.reserve(std::size_t(dim()));
    for (int64_t d = 0; d < dim(); ++d) pieces.push_back(aux_split_dim(*this, I, d));
    return pieces;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
