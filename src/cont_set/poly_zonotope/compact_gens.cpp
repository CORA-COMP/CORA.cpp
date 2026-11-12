// compact_gens - equal exponents of a polynomial zonotope added, nothing else dropped
//
// The dependent generators whose exponent columns are equal are added into the column of the least
// index. The sum of a coordinate runs over all columns from the last to the first, a column of
// another class adding 0 (CORALean's fiberDown, a masked sum). The factors, the positions of the
// kept columns and the independent generators stay, no zero generator is dropped, and a column with
// all exponents 0 stays a generator. Unlike the merge of the constructor (CORA's
// removeRedundantExponents) and compact, which fold the constants and drop the zero columns.
//
// Syntax:   S = pZ.compact_gens();
// Inputs:   -
// Outputs:  S - the same set with one column per distinct exponent column
// See also: compact
//
// CORA.cpp only: the merge of CORALean's PolyZonotope.compactGens, which keeps every column.
// Authors:       Claude Sonnet 5.5
// Written:       12-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"

#include <map>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

PolyZonotope PolyZonotope::compact_gens() const {
    detail::Entry entry;
    const int64_t n = dim(), h = num_dependent();
    // the least column of each exponent column
    std::map<std::vector<int>, int64_t> first;
    std::vector<int64_t> rep(static_cast<std::size_t>(h)), kept;
    for (int64_t j = 0; j < h; ++j) {
        const auto [at, inserted] = first.emplace(E_[std::size_t(j)], j);
        rep[std::size_t(j)] = at->second;
        if (inserted) kept.push_back(j);
    }
    if (int64_t(kept.size()) == h) return *this;
    // the sums of the classes, each from the last member to the first (masked zeros add nothing)
    const std::vector<double> g = G_.data();
    const auto k = kept.size();
    std::vector<std::vector<int64_t>> members(k);
    std::vector<std::size_t> slot(static_cast<std::size_t>(h));
    for (std::size_t t = 0; t < k; ++t) slot[std::size_t(kept[t])] = t;
    for (int64_t l = 0; l < h; ++l) members[slot[std::size_t(rep[std::size_t(l)])]].push_back(l);
    std::vector<double> merged(std::size_t(n) * k);
    Exponents E;
    for (std::size_t t = 0; t < k; ++t) {
        E.push_back(E_[std::size_t(kept[t])]);
        for (int64_t i = 0; i < n; ++i) {
            double sum = 0;
            for (std::size_t m = members[t].size(); m-- > 0;)
                sum = g[std::size_t(i * h + members[t][m])] + sum;
            merged[std::size_t(i) * k + t] = sum;
        }
    }
    return {
        c_, Tensor::like(G_, merged, {n, int64_t(k)}), GI_, std::move(E), id_, detail::Verbatim{}};
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
