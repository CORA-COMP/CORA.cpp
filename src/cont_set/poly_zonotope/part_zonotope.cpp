// part_zonotope - a polynomial zonotope grouped by exponent patterns, as CORA's partZonotope
//
// The dependent generators are grouped by their pattern of exponents in the factors ids. Every
// group except the zero pattern gets a new factor t that stands for the monomial of the chosen
// factors, which lies in [-1, 1]. For a pattern of only even exponents the monomial lies in [0, 1]
// and is (1 + t)/2: the generators are halved and a copy of the halves without t is added behind
// the others (CORA's pZevenShift). The chosen factors stay in the set with zero exponents. The
// groups are in the order of the last occurrence of their pattern (Mathlib's dedup, the order of
// CORALean's partPatterns); center and independent generators are copied. CORA's grouping outputs
// are not returned (docs/cora-differences.md).
//
// Syntax:   S = pZ.part_zonotope(ids);
// Inputs:   ids - identifiers of the factors that are grouped, all of them in the set
// Outputs:  S - a polynomial zonotope that contains the set, with one new factor per pattern
// See also: compact, restructure
//
// CORA: contSet/@polyZonotope/partZonotope.m
// CORA authors: Niklas Kochdumper, Tobias Ladner
// Authors:       Claude Sonnet 5.5
// Written:       23-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "global/format.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

PolyZonotope PolyZonotope::part_zonotope(const std::vector<int64_t> &ids) const {
    detail::Entry entry;
    const std::size_t p = id_.size();
    std::vector<bool> chosen(p, false);
    for (const int64_t id : ids) {
        const int64_t at = priv_find_id(id_, id);
        CORA_CHECK(at >= 0, Error::wrong_value, [&] {
            return "part_zonotope: the identifier " + std::to_string(id) + " is not in the set";
        });
        chosen[std::size_t(at)] = true;
    }
    // the pattern of every generator in the chosen factors
    const int64_t h = num_dependent();
    std::vector<std::vector<int>> pattern(std::size_t(h), std::vector<int>(p, 0));
    for (int64_t j = 0; j < h; ++j)
        for (std::size_t k = 0; k < p; ++k)
            if (chosen[k]) pattern[std::size_t(j)][k] = E_[std::size_t(j)][k];
    // the distinct patterns, each at the place of its last occurrence
    std::vector<std::vector<int>> groups;
    for (int64_t j = 0; j < h; ++j) {
        const auto it = std::find(groups.begin(), groups.end(), pattern[std::size_t(j)]);
        if (it != groups.end()) groups.erase(it);
        groups.push_back(pattern[std::size_t(j)]);
    }
    const std::size_t g = groups.size();
    std::vector<int64_t> id = priv_append_fresh_ids(id_, int64_t(g));
    // which generators are halved: only even exponents and not all zero
    std::vector<double> halve(std::size_t(h), 1.0),
        shifted(std::size_t(h) * std::size_t(dim()), 0.0);
    // the exponent columns of the two blocks, the new factor in the first
    Exponents first_block, second_block;
    // per generator: halved or not, the exponent columns of both blocks
    for (int64_t j = 0; j < h; ++j) {
        const std::vector<int> &pat = pattern[std::size_t(j)];
        const bool zero = std::all_of(pat.begin(), pat.end(), [](int e) { return e == 0; });
        const bool even = std::all_of(pat.begin(), pat.end(), [](int e) { return e % 2 == 0; });
        halve[std::size_t(j)] = even && !zero ? 0.5 : 1.0;
        if (even && !zero) std::fill_n(shifted.begin() + j * dim(), dim(), 1.0);
        std::vector<int> e = E_[std::size_t(j)];
        for (std::size_t k = 0; k < p; ++k)
            if (chosen[k]) e[k] = 0;
        // the first block's column: the new factor of the pattern
        std::vector<int> col = e;
        col.resize(p + g, 0);
        if (!zero) {
            const auto at = std::find(groups.begin(), groups.end(), pat);
            col[p + std::size_t(at - groups.begin())] = 1;
        }
        // the second block's column keeps the old factors only
        first_block.push_back(std::move(col));
        e.resize(p + g, 0);
        second_block.push_back(std::move(e));
    }
    Exponents E = std::move(first_block);
    E.insert(E.end(), second_block.begin(), second_block.end());
    // the halves, then a copy where the generator is shifted, +0 elsewhere
    const Tensor w = Tensor::like(G_, halve, {h, 1}), mask = Tensor::like(G_, shifted, {h, dim()});
    const Tensor halves = G_.multiply(w);
    return {c_,
            Tensor::concat({halves, Tensor::where(mask, halves, G_.zeros_like())}, -2),
            GI_,
            std::move(E),
            std::move(id),
            detail::Verbatim{}};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
