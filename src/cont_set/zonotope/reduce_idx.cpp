// reduce_idx - a zonotope with the generators of given indices replaced by their box
//
// The chosen generators leave and the box around them, n axis generators, is appended; the
// others keep their place and order. The tuning of an adaptive reachability analysis chooses the
// indices once per step (reduce_adaptive) and applies them to every pass of the step. No indices:
// the zonotope as it is.
//
// Syntax:   Zred = Z.reduce_idx(idx);
// Inputs:   idx - 0-based indices of generators of Z
// Outputs:  Zred - a zonotope that contains Z
// See also: reduce, reduce_adaptive
//
// CORA: contSet/@zonotope/private/priv_reduceIdx.m
// CORA authors: Mark Wetzlinger
// Authors:       Claude Sonnet 5.5
// Written:       10-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/zonotope/private/priv.h"
#include "cora/cont_set/zonotope/zonotope.h"
#include "cora/global/error.h"

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

Zonotope Zonotope::reduce_idx(const std::vector<int64_t> &idx) const {
    detail::Entry entry;
    CORA_CHECK(c_.ndim() == 2 && G_.ndim() == 2, Error::not_supported,
               [] { return std::string("reduce_idx: a batch of zonotopes is not supported"); });
    if (idx.empty()) return *this;
    const int64_t m = G_.shape().back();
    std::vector<char> boxed(std::size_t(m), 0);
    for (const int64_t j : idx) {
        CORA_CHECK(j >= 0 && j < m, Error::wrong_value, [&] {
            return "reduce_idx: the index " + std::to_string(j) + " exceeds the " +
                   std::to_string(m) + " generators";
        });
        boxed[std::size_t(j)] = 1;
    }
    std::vector<int64_t> kept;
    for (int64_t j = 0; j < m; ++j)
        if (!boxed[std::size_t(j)]) kept.push_back(j);
    // the box of the chosen generators, summed in the order of idx, after the kept ones
    const Tensor diag = G_.take(idx, -1).abs().sum(-1, true).diag();
    if (kept.empty()) return {c_, diag};
    return {c_, Tensor::concat({G_.take(kept, -1), diag}, -1)};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
