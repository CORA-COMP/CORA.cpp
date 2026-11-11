// split_longest_gen - the halves of a polynomial zonotope along the factor of its longest generator
//
// The longest dependent generator is the one of the largest sum of squares (the first on a tie);
// its factor is the one of its largest exponent (the first on a tie). The set is then split along
// that factor (split_dep_factor).
//
// Syntax:   [S1, S2] = S.split_longest_gen();
// Inputs:   -
// Outputs:  S1, S2 - the halves, which cover the set
// See also: split_dep_factor
//
// CORA: contSet/@polyZonotope/splitLongestGen.m
// CORA authors: Niklas Kochdumper
// Authors:       Claude Sonnet 5.5
// Written:       11-November-2026
// Last update:   ---
// Reviewed:      ---

#include "cont_set/poly_zonotope/private/priv.h"
#include "global/format.h"

#include <algorithm>

// ----------------------------------------  BEGIN CODE  ---------------------------------------- //

namespace cora {


// ===========================================  MAIN  =========================================== //

std::pair<PolyZonotope, PolyZonotope> PolyZonotope::split_longest_gen() const {
    detail::Entry entry;
    CORA_CHECK(num_dependent() > 0, Error::wrong_value,
               [] { return std::string("split_longest_gen: the set has no dependent generator"); });
    // the longest generator by the sum of squares of its entries
    const int64_t n = dim(), h = num_dependent();
    const std::vector<double> G = G_.data();
    std::vector<double> sq(static_cast<std::size_t>(h), 0.0);
    for (int64_t j = 0; j < h; ++j)
        for (int64_t r = 0; r < n; ++r)
            sq[std::size_t(j)] += G[std::size_t(r * h + j)] * G[std::size_t(r * h + j)];
    const auto longest = std::size_t(std::max_element(sq.begin(), sq.end()) - sq.begin());
    // its factor of the largest exponent
    const std::vector<int> &e = E_[longest];
    const auto row = std::size_t(std::max_element(e.begin(), e.end()) - e.begin());
    return split_dep_factor(id_[row]);
}

} // namespace cora

// ---------------------------------------  END OF CODE  ---------------------------------------- //
