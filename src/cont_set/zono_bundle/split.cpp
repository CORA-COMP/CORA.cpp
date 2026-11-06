// split - the two halves of a zonotope bundle cut at a coordinate, as CORA's zonoBundle.split
//
// The box around the bundle is cut at the middle of the coordinate; each half of the box is
// intersected with the bundle (a member more). CORA's directional forms (a halfspace, a direction
// with "bundle") are not ported. Not a batch.
//
// Syntax:   auto [left, right] = bundle.split(N);   auto pieces = bundle.split();
// Inputs:   N - the coordinate (0-based)
// Outputs:  left, right - the pieces, which cover the bundle; split() has the pair of every
// coordinate See also: ZonoBundle::intersect, ConZonotope::split
//
// CORA: contSet/@zonoBundle/split.m
// CORA authors: Matthias Althoff
// Authors:       Claude Sonnet 5.5
// Written:       06-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/cont_set/private/priv_coordinate.h"
#include "cont_set/zono_bundle/private/priv.h"
#include "cora/cont_set/interval/interval.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {

// ----------------------------------------  AUXILIARY  ----------------------------------------- //

namespace {

/// Why the split is refused, if it is: a batch, or an empty set.
detail::Refusal aux_refusal(const ZonoBundle &bundle) {
    if (bundle.zonotopes().front().center().ndim() != 2)
        return std::pair{Error::not_supported,
                         std::string("split: it supports a single set, not a batch")};
    if (priv_bundle_is_empty(bundle))
        return std::pair{Error::empty_set, std::string("split: the set is empty")};
    return std::nullopt;
}

/// The pieces of the box I cut at the middle of coordinate d (CORA's aux_splitOneDim): the lower
/// half has the middle as its upper end there, the upper half as its lower end.
std::pair<ZonoBundle, ZonoBundle> aux_split_dim(const ZonoBundle &bundle, const Interval &I,
                                                int64_t d) {
    const Tensor lower = I.inf(), upper = I.sup();
    std::vector<double> unit(std::size_t(bundle.dim()), 0.0);
    unit[std::size_t(d)] = 1.0;
    const Tensor at = Tensor::like(lower, unit, {bundle.dim(), 1});
    const Tensor middle = (lower + upper).multiply(at) * 0.5, keep = at * -1.0 + 1.0;
    const Tensor lower_end = upper.multiply(keep) + middle,
                 upper_start = lower.multiply(keep) + middle;
    return {bundle.intersect(Interval(lower, lower_end)),
            bundle.intersect(Interval(upper_start, upper))};
}

} // namespace


// ===========================================  MAIN  =========================================== //

std::pair<ZonoBundle, ZonoBundle> ZonoBundle::split(int64_t N) const {
    detail::Entry entry;
    detail::Refusal refusal = priv_coordinate_refusal(N, dim(), "split");
    if (!refusal) refusal = aux_refusal(*this);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    return aux_split_dim(*this, interval(), N);
}

std::vector<std::pair<ZonoBundle, ZonoBundle>> ZonoBundle::split() const {
    detail::Entry entry;
    const detail::Refusal refusal = aux_refusal(*this);
    CORA_CHECK(!refusal, refusal->first, [&] { return refusal->second; });
    std::vector<std::pair<ZonoBundle, ZonoBundle>> pieces;
    // the pair of each coordinate, each cut from the box of the bundle
    for (int64_t d = 0; d < dim(); ++d) pieces.push_back(split(d));
    return pieces;
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
